import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from ReferenceNet import ReferenceNet
from diffusers import DDIMScheduler
import argparse
from inpainting_pipeline import StableDiffusionInpaintPipeline
from huggingface_hub import snapshot_download
import numpy as np
import math
import os
import cv2


class LinearResampler(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024):
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


# ===================== Model Setup =====================

parser = argparse.ArgumentParser(description="Gradio Demo")

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
def generate(background_image_path, mask_path, ref_image_path):
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
    generated_images = [img.resize(target_image_size) for img in generated_images]
    return generated_images[0]


# --- Visualization Processor ---

class CaptureAttnProcessor(nn.Module):
    def __init__(self):
        self.captured_attn_map = None
        self.cnt = 0
        super().__init__()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.cnt % 3 == 0:
            if attention_probs.shape[0] > 8:
                self.captured_attn_map = attention_probs[8:, :, :].detach()
            else:
                self.captured_attn_map = attention_probs.detach()

        self.cnt += 1

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, 2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



def visualize_attention_map(hooks_dict, save_path, inpainting_mask=None, ref_image=None):
    valid_hooks = {k: v for k, v in hooks_dict.items() if v.captured_attn_map is not None}
    num_layers = len(valid_hooks)

    if num_layers == 0:
        print("No attention maps captured.")
        return

    sorted_keys = sorted(valid_hooks.keys())
    layer_name = sorted_keys[0]
    proc = valid_hooks[layer_name]
    attention_map = proc.captured_attn_map

    attn_avg = attention_map.mean(dim=0).cpu().detach()

    split_idx = int(attn_avg.shape[0] / 2)
    height = width = int(math.sqrt(split_idx))

    attn_cross = attn_avg[:split_idx, split_idx:]

    if inpainting_mask is not None:
        mask_resized = inpainting_mask.resize((width, height), resample=Image.NEAREST)
        mask_array = np.array(mask_resized).astype(np.float32)

        if mask_array.max() > 1.0:
            mask_array = mask_array / 255.0

        mask_binary = mask_array > 0.5
        mask_flat = mask_binary.flatten()

        num_mask_pixels = int(mask_flat.sum())
        if num_mask_pixels > 0:
            attn_mask_region = attn_cross[mask_flat, :]
            map_data = attn_mask_region.mean(dim=0).reshape(height, width).numpy()
            print(f"Using {num_mask_pixels}/{len(mask_flat)} query positions from mask region")
        else:
            print("Warning: mask region is empty after binarization, falling back to full attention map")
            map_data = attn_cross.mean(dim=0).reshape(height, width).numpy()
    else:
        map_data = attn_cross.mean(dim=0).reshape(height, width).numpy()

    map_data = (map_data - map_data.min()) / (map_data.max() - map_data.min() + 1e-8)
    heatmap = (map_data * 255).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    target_size = (512, 512)
    heatmap_img = cv2.resize(heatmap_img, target_size)

    if ref_image is not None:
        org_img = np.array(ref_image.resize(target_size, resample=Image.BICUBIC))
        org_img = cv2.resize(org_img, target_size)
        attn_vis = cv2.addWeighted(org_img, 0.3, heatmap_img, 0.7, 0)
    else:
        attn_vis = heatmap_img

    cv2.imwrite(save_path, cv2.cvtColor(attn_vis, cv2.COLOR_RGB2BGR))
    print(f"Saved {'overlayed' if ref_image is not None else 'standalone'} "
          f"JET heatmap (512x512) for {layer_name} to: {save_path}")


class AttentionVisualizer:
    def __init__(self, pipe):
        self.pipe = pipe
        self.hooks = {}

    def register_specific_layer(self):
        unet = self.pipe.unet

        target_block_idx = 3
        target_attn_idx = 2

        try:
            block = unet.up_blocks[target_block_idx]

            if hasattr(block, "attentions") and len(block.attentions) > target_attn_idx:
                attn_module = block.attentions[target_attn_idx]

                for k, transformer in enumerate(attn_module.transformer_blocks):
                    target_attn = transformer.attn1
                    layer_name = f"up_blocks.{target_block_idx}.attentions.{target_attn_idx}"

                    hook_proc = CaptureAttnProcessor()
                    target_attn.set_processor(hook_proc)
                    self.hooks[layer_name] = hook_proc
                    print(f"Successfully registered hook: {layer_name}")
            else:
                print(f"Error: Layer up_blocks.{target_block_idx}.attentions.{target_attn_idx} does not exist.")

        except IndexError:
            print(f"Error: up_blocks index {target_block_idx} out of range.")

    def visualize(self, save_path, inpainting_mask=None, ref_image=None):
        visualize_attention_map(self.hooks, save_path,
                                inpainting_mask=inpainting_mask,
                                ref_image=ref_image)


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    background_image_path = "validation_images/background_image_2.png"
    mask_path = "validation_images/inpainting_mask_2.png"
    ref_image_path = "validation_images/ref_image_2.png"

    visualizer = AttentionVisualizer(reference_anomaly_model.pipe)
    visualizer.register_specific_layer()

    generate(background_image_path, mask_path, ref_image_path)

    save_path = "visualization_results/reference_image_attention_map.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pil_mask_image = Image.open(mask_path).convert("L").resize((512, 512))
    pil_ref_image = Image.open(ref_image_path).convert("RGB").resize((512, 512))

    visualizer.visualize(save_path, inpainting_mask=pil_mask_image, ref_image=pil_ref_image)
