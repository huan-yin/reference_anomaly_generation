import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModelWithProjection
from ReferenceNet import ReferenceNet
from diffusers import DDIMScheduler
from inpainting_pipeline import StableDiffusionInpaintPipeline
import argparse
from huggingface_hub import snapshot_download

  
class LinearResampler(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        output_dim=1024,
    ):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projector(x)
    
class ReferencenetInpainting:
    def __init__(self, sd_pipe, referencenet, image_encoder_path, checkpoint_path, device):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.checkpoint_path = checkpoint_path
        self.referencenet = referencenet.to(self.device)
        self.pipe = sd_pipe.to(self.device)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )

        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = self.init_proj()
        self.load_unet_and_image_proj_and_referencenet()

    def init_proj(self):
        image_proj_model = LinearResampler(
            input_dim=1280,
            output_dim=self.pipe.unet.config.cross_attention_dim,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def load_unet_and_image_proj_and_referencenet(self):
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        self.pipe.unet.load_state_dict(state_dict["unet"], strict=False)
        self.referencenet.load_state_dict(state_dict["referencenet"], strict=False)
        self.image_proj_model.load_state_dict(state_dict["image_proj"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values

        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]

        image_prompt_embeds = self.image_proj_model(clip_image_embeds).to(dtype=torch.float16)

        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_ref_image=None,
        pil_background_image=None,
        pil_mask_image=None,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image=pil_ref_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        images = self.pipe(
            image=pil_background_image,
            mask_image=pil_mask_image,
            prompt_embeds=image_prompt_embeds,
            negative_prompt_embeds=uncond_image_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            referencenet=self.referencenet,
            ref_image=pil_ref_image,
            clip_image_embed=torch.cat([uncond_image_prompt_embeds, image_prompt_embeds], dim=0),
            **kwargs,
        ).images

        return images




parser = argparse.ArgumentParser(description="Inference script for reference-based anomaly generation")

allow_sd_text_encoder_patterns = ["text_encoder/config.json", "text_encoder/pytorch_model.bin"]
allow_tokenizer_patterns = ["tokenizer/*"]
allow_scheduler_patterns = ["scheduler/*"]
allow_vae_patterns = ["vae/config.json", "vae/diffusion_pytorch_model.bin"]
allow_unet_patterns = ["unet/config.json", "unet/diffusion_pytorch_model.bin"]
allow_sd_patterns = allow_sd_text_encoder_patterns + allow_tokenizer_patterns + allow_scheduler_patterns + allow_vae_patterns + allow_unet_patterns + ["model_index.json"]

sd_model_path = snapshot_download("stable-diffusion-v1-5/stable-diffusion-inpainting", allow_patterns=allow_sd_patterns)
ref_model_path = snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", allow_patterns=allow_sd_patterns)
image_encoder_path = snapshot_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", allow_patterns=["config.json", "pytorch_model.bin"])
checkpoint_path = snapshot_download('LiXiY/ReferenceAnomaly') + "/" + "reference_anomaly_checkponint.bin"

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parser.parse_args()

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    sd_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    feature_extractor=None,
    safety_checker=None
)

referencenet = ReferenceNet.from_pretrained(ref_model_path, subfolder="unet", feature_extractor=None, safety_checker=None).to(dtype=torch.float16)

reference_anomaly_model = ReferencenetInpainting(pipe, referencenet, image_encoder_path, checkpoint_path, device)
    

@torch.no_grad()
def generate(ref_image_path, background_image_path, mask_path, save_dir):
   
    os.makedirs(save_dir, exist_ok=True)
    target_image_size = (512, 512)  


    pil_ref_image = Image.open(ref_image_path).convert("RGB").resize(target_image_size)
    pil_background_image = Image.open(background_image_path).convert("RGB").resize(target_image_size)
    pil_mask_image = Image.open(mask_path).convert("L").resize(target_image_size)

    
 
    generated_images = reference_anomaly_model.generate(
        pil_ref_image=pil_ref_image,
        pil_background_image=pil_background_image,
        pil_mask_image=pil_mask_image,
        num_samples=1,
        num_inference_steps=20,
        seed=42
    )

    generated_images[0].resize(target_image_size).save(os.path.join(save_dir, "inference_image.png"))

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="visualization_results",
        help="Path to save generated images",
    )
    parser.add_argument(
        "--background_image_path",
        type=str,
        default="validation_images/background_image_4.png",
        help="Path to background image",
    )
    parser.add_argument(
        "--inpainting_mask_path",
        type=str,
        default="validation_images/inpainting_mask_4.png",
        help="Path to inpainting mask",
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default="validation_images/ref_image_4.png",
        help="Path to reference image",
    )


    
    args = parser.parse_args()
   
    return args


if __name__ == "__main__":
    args = parse_args()
    generate(
        args.ref_image_path,
        args.background_image_path,
        args.inpainting_mask_path,
        args.save_dir,
    )

    

    