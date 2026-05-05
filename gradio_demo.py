import argparse
import base64
import os
from io import BytesIO
import gradio as gr
import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from ReferenceNet import ReferenceNet

from inpainting_pipeline import StableDiffusionInpaintPipeline
from PIL import Image
from huggingface_hub import snapshot_download
import cv2
import numpy as np
import math


class LinearResampler(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projector(x)


# ===================== Attention Capture =====================

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


def visualize_attention_map(hooks_dict, inpainting_mask=None, ref_image=None):
    """
    Visualize the attention map and return a PIL Image (without saving to disk).
    Uses cv2 COLORMAP_JET heatmap overlaid on the reference image.
    """
    valid_hooks = {k: v for k, v in hooks_dict.items() if v.captured_attn_map is not None}
    num_layers = len(valid_hooks)

    if num_layers == 0:
        print("No attention maps captured.")
        return None

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

    # Normalize to 0–255
    map_data = (map_data - map_data.min()) / (map_data.max() - map_data.min() + 1e-8)
    heatmap = (map_data * 255).astype(np.uint8)

    # JET pseudo-color mapping
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    target_size = (512, 512)
    heatmap_img = cv2.resize(heatmap_img, target_size)

    # Overlay onto the reference image
    if ref_image is not None:
        org_img = np.array(ref_image.resize(target_size, resample=Image.BICUBIC))
        org_img = cv2.resize(org_img, target_size)
        attn_vis = cv2.addWeighted(org_img, 0.3, heatmap_img, 0.7, 0)
    else:
        attn_vis = heatmap_img

    return Image.fromarray(attn_vis)


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

    def reset(self):
        """Reset the capture state of all hooks. Should be called before each generation."""
        for proc in self.hooks.values():
            proc.captured_attn_map = None
            proc.cnt = 0

    def visualize(self, inpainting_mask=None, ref_image=None):
        """Return a PIL Image visualization of the attention map."""
        return visualize_attention_map(
            self.hooks, inpainting_mask=inpainting_mask, ref_image=ref_image
        )


# ===================== Model Wrapper =====================

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

# ===================== Register Attention Hook =====================
attention_visualizer = AttentionVisualizer(reference_anomaly_model.pipe)
attention_visualizer.register_specific_layer()


# ===================== Example Data =====================

CANVAS_W, CANVAS_H = 512, 512

EXAMPLES = [
    ("validation_images/background_image_1.png", "validation_images/ref_image_1.png", "validation_images/inpainting_mask_1.png"),
    ("validation_images/background_image_2.png", "validation_images/ref_image_2.png", "validation_images/inpainting_mask_2.png"),
    ("validation_images/background_image_3.png", "validation_images/ref_image_3.png", "validation_images/inpainting_mask_3.png"),
    ("validation_images/background_image_4.png", "validation_images/ref_image_4.png", "validation_images/inpainting_mask_4.png"),
]


# ===================== Thumbnail HTML Generation =====================

def img_to_b64(path, size=(120, 120)):
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_examples_html():
    row_pairs = [EXAMPLES[0:2], EXAMPLES[2:4]]

    html = '<div class="ex-grid">'
    for row_idx, row_examples in enumerate(row_pairs):
        html += '<div class="ex-grid-row">'
        for col_idx, (bg, ref, mask) in enumerate(row_examples):
            i = row_idx * 2 + col_idx
            bg_b64 = img_to_b64(bg)
            ref_b64 = img_to_b64(ref)
            mask_b64 = img_to_b64(mask)
            html += f'''
            <div class="ex-row" onclick="(function(){{var el=document.getElementById('ex_btn_{i}');if(!el)return;var btn=el.querySelector('button')||el;btn.dispatchEvent(new MouseEvent('click',{{bubbles:true,cancelable:true}}));}})()">
                <div class="ex-label">Example {i + 1}</div>
                <div class="ex-thumbs">
                    <div class="ex-thumb-wrap">
                        <img src="data:image/png;base64,{bg_b64}" class="ex-thumb" draggable="false"/>
                        <span class="ex-thumb-sublabel">Background</span>
                    </div>
                    <div class="ex-thumb-wrap">
                        <img src="data:image/png;base64,{mask_b64}" class="ex-thumb" draggable="false"/>
                        <span class="ex-thumb-sublabel">Mask</span>
                    </div>
                    <div class="ex-thumb-wrap">
                        <img src="data:image/png;base64,{ref_b64}" class="ex-thumb" draggable="false"/>
                        <span class="ex-thumb-sublabel">Reference</span>
                    </div>
                </div>
            </div>
            '''
        html += '</div>'
    html += '</div>'
    return html


# ===================== Utility Functions =====================

def fit_image_to_canvas(img, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    img_rgba = img.convert("RGBA")
    img_rgba.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    offset_x = (canvas_w - img_rgba.width) // 2
    offset_y = (canvas_h - img_rgba.height) // 2
    canvas.paste(img_rgba, (offset_x, offset_y))
    return canvas


def extract_mask_from_layers(layers, target_size):
    mask = Image.new("L", target_size, 0)
    for layer in layers:
        if layer is not None:
            layer_rgba = layer.convert("RGBA").resize(target_size)
            alpha = layer_rgba.split()[3]
            alpha_binary = alpha.point(lambda x: 255 if x > 0 else 0)
            mask = Image.composite(Image.new("L", target_size, 255), mask, alpha_binary)
    return mask


def load_example(idx):
    bg_path, ref_path, mask_path = EXAMPLES[idx]

    bg = Image.open(bg_path).convert("RGB").resize((CANVAS_W, CANVAS_H))
    ref_img = Image.open(ref_path).convert("RGB").resize((CANVAS_W, CANVAS_H))
    mask = Image.open(mask_path).convert("L").resize((CANVAS_W, CANVAS_H))

    transparent = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    white_solid = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 255))
    mask_layer = Image.composite(white_solid, transparent, mask)

    composite = Image.alpha_composite(bg.convert("RGBA"), mask_layer)

    editor_val = {
        "background": bg,
        "layers": [mask_layer],
        "composite": composite,
    }
    return editor_val, ref_img


def load_ex1():
    return load_example(0)

def load_ex2():
    return load_example(1)

def load_ex3():
    return load_example(2)

def load_ex4():
    return load_example(3)


# ===================== Generation Function (also returns attention map) =====================

def run_local(base, ref):
    if base is None or ref is None:
        return None, None, gr.update(visible=False)

    target_size = (CANVAS_W, CANVAS_H)
    pil_ref = ref.convert("RGB").resize(target_size)

    if not isinstance(base, dict):
        return None, None, gr.update(visible=False)

    bg_pil = base.get("background")
    layers = base.get("layers", [])

    if bg_pil is None:
        return None, None, gr.update(visible=False)

    pil_bg = bg_pil.convert("RGB").resize(target_size)
    pil_mask = extract_mask_from_layers(layers, target_size)

    if pil_mask.getextrema() == (0, 0):
        error_html = """
            <div class="error-overlay" style="
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.5); display: flex; justify-content: center;
                align-items: center; z-index: 9999;
            ">
                <div style="
                    background: white; padding: 30px; border-radius: 10px;
                    text-align: center; font-size: 18px; box-shadow: 0 0 15px rgba(0,0,0,0.3);
                ">
                    <p style="color: red; margin-bottom: 20px;">
                        ⚠️ Please draw the anomaly region (mask) on the background image first, or click an example!
                    </p>
                    <button onclick="this.closest('.error-overlay').remove()"
                            style="padding: 8px 20px; cursor: pointer; border: none;
                                background: #eee; border-radius: 5px;">
                        OK
                    </button>
                </div>
            </div>
            """
        return None, None, gr.update(value=error_html, visible=True)

    # Reset attention capture before generation
    attention_visualizer.reset()

    generated_images = reference_anomaly_model.generate(
        pil_ref_image=pil_ref,
        pil_background_image=pil_bg,
        pil_mask_image=pil_mask,
        num_samples=1,
        guidance_scale=7.5,
        num_inference_steps=25,
        seed=42,
    )

    result_img = generated_images[0].resize(target_size)

    # Generate attention map visualization
    attn_img = attention_visualizer.visualize(
        inpainting_mask=pil_mask,
        ref_image=pil_ref,
    )

    return result_img, attn_img, gr.update(visible=False)


# ===================== Combined Client JS (resize + force English) =====================
# KEY FIX: merge both JS functions into a SINGLE function body instead of
# concatenating two separate function expressions.

COMBINED_JS = """
function() {
    /* ===== Client-Side Instant Resize ===== */
    var MAX_W = """ + str(CANVAS_W) + """;
    var MAX_H = """ + str(CANVAS_H) + """;

    function resizeInBrowser(file) {
        return new Promise(function(resolve) {
            var reader = new FileReader();
            reader.onload = function(e) {
                var img = new Image();
                img.onload = function() {
                    if (img.width <= MAX_W && img.height <= MAX_H) {
                        resolve(null);
                        return;
                    }
                    var ratio = Math.min(MAX_W / img.width, MAX_H / img.height);
                    var c = document.createElement('canvas');
                    c.width  = Math.round(img.width  * ratio);
                    c.height = Math.round(img.height * ratio);
                    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
                    c.toBlob(function(blob) {
                        resolve(blob ? new File([blob], file.name, {type: 'image/png'}) : null);
                    }, 'image/png');
                };
                img.onerror = function() { resolve(null); };
                img.src = e.target.result;
            };
            reader.onerror = function() { resolve(null); };
            reader.readAsDataURL(file);
        });
    }

    function hookInput(inp) {
        if (inp._resizeHooked) return;
        inp._resizeHooked = true;
        inp._skipResize  = false;

        inp.addEventListener('change', function(e) {
            if (inp._skipResize) { inp._skipResize = false; return; }

            var file = inp.files && inp.files[0];
            if (!file || !file.type || file.type.indexOf('image/') !== 0) return;

            e.stopImmediatePropagation();
            e.stopPropagation();

            resizeInBrowser(file).then(function(resized) {
                if (resized) {
                    var dt = new DataTransfer();
                    dt.items.add(resized);
                    inp.files = dt.files;
                }
                inp._skipResize = true;
                inp.dispatchEvent(new Event('change', {bubbles: true}));
            });
        }, true);
    }

    function scan() {
        var inputs = document.querySelectorAll('.input-row input[type="file"]');
        for (var i = 0; i < inputs.length; i++) hookInput(inputs[i]);
    }

    new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
    scan();

    /* ===== Force English UI Labels ===== */
    var zh2en = {
        '将图像拖放到此处或点击上传': 'Drag image here or click to upload',
        '拖放文件到这里': 'Drag file here',
        '点击上传': 'Click to upload',
        '或点击上传': 'or click to upload',
        '上传图片': 'Upload image',
        '粘贴图片或URL': 'Paste image or URL',
        '清空': 'Clear',
        '编辑': 'Edit',
        '撤销': 'Undo',
        '重做': 'Redo',
        '缩放': 'Zoom',
        '画笔': 'Brush',
        '橡皮擦': 'Eraser',
        '清除图层': 'Clear layers',
        '图像编辑器': 'Image Editor',
        '生成': 'Generate',
        '正在运行...': 'Running...',
        '提交': 'Submit',
    };

    function translateNode(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            var text = node.textContent;
            for (var zh in zh2en) {
                if (text.indexOf(zh) !== -1) {
                    text = text.split(zh).join(zh2en[zh]);
                }
            }
            if (text !== node.textContent) node.textContent = text;
        }
    }

    function walkAndTranslate(root) {
        var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
        var node;
        while (node = walker.nextNode()) translateNode(node);
    }

    function translateAttributes(root) {
        root.querySelectorAll('[placeholder]').forEach(function(el) {
            var ph = el.getAttribute('placeholder');
            for (var zh in zh2en) {
                if (ph.indexOf(zh) !== -1) ph = ph.split(zh).join(zh2en[zh]);
            }
            el.setAttribute('placeholder', ph);
        });
        root.querySelectorAll('[title]').forEach(function(el) {
            var t = el.getAttribute('title');
            for (var zh in zh2en) {
                if (t.indexOf(zh) !== -1) t = t.split(zh).join(zh2en[zh]);
            }
            el.setAttribute('title', t);
        });
    }

    function runTranslate() {
        walkAndTranslate(document.body);
        translateAttributes(document.body);
    }

    var translateObserver = new MutationObserver(function(mutations) {
        for (var m = 0; m < mutations.length; m++) {
            var added = mutations[m].addedNodes;
            for (var n = 0; n < added.length; n++) {
                if (added[n].nodeType === Node.ELEMENT_NODE) {
                    walkAndTranslate(added[n]);
                    translateAttributes(added[n]);
                }
            }
        }
    });

    translateObserver.observe(document.body, { childList: true, subtree: true });
    runTranslate();
    setInterval(runTranslate, 2000);
}
"""


# ===================== Gradio UI =====================

with gr.Blocks(css="""
    .input-row {
        overflow: visible !important;
    }

    .input-row .gr-image-editor {
        overflow: hidden !important;
    }
    .input-row .gr-image-editor .image-container,
    .input-row .gr-image-editor .canvas-container,
    .input-row .gr-image-editor canvas {
        max-width: 100% !important;
        max-height: 100% !important;
        object-fit: contain !important;
    }

    .ex-section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 28px 0 14px 0;
        justify-content: center;
    }
    .ex-section-header::before {
        content: '';
        flex: 1;
        height: 1px;
        max-width: 180px;
        background: #e5e7eb;
    }
    .ex-section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        max-width: 180px;
        background: #e5e7eb;
    }

    .ex-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-bottom: 20px;
    }

    .ex-grid {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        padding-bottom: 20px;
    }
    .ex-grid-row {
        display: flex;
        gap: 20px;
        justify-content: center;
        flex-wrap: wrap;
    }

    .ex-row {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 14px 28px;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.25s ease;
        background: #ffffff;
        user-select: none;
        width: fit-content;
    }
    .ex-row:hover {
        border-color: #3b82f6;
        background: #f0f7ff;
        box-shadow: 0 4px 18px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    .ex-row:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
    }

    .ex-label {
        font-weight: 700;
        font-size: 15px;
        min-width: 62px;
        color: #1e40af;
        letter-spacing: 0.02em;
    }

    .ex-thumbs {
        display: flex;
        gap: 14px;
    }

    .ex-thumb-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
    }

    .ex-thumb {
        width: 110px;
        height: 110px;
        object-fit: cover;
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: all 0.25s ease;
        pointer-events: none;
    }
    .ex-row:hover .ex-thumb {
        border-color: #93c5fd;
    }

    .ex-thumb-sublabel {
        font-size: 12px;
        color: #6b7280;
        font-weight: 500;
    }
""", js=COMBINED_JS) as demo:

    gr.Markdown(
        "<h1 style='text-align: center;'>Reference-Based Anomaly Image Generation</h1>"
        "<h3 style='text-align: center;'>Generate anomaly images similar to the reference anomaly on normal images</h3>"
    )
    gr.Markdown(
        """
        **Instructions:**
        1. Upload a background image (normal object), then use the brush tool below the image to mark the region where you want to generate an anomaly (mask), and upload a reference image (reference anomaly).
        2. Or click any row of thumbnails in the "Examples" section below to automatically load a background + mask + reference image.
        3. Click the "Generate" button, and the result will be displayed below.
        """
    )

    with gr.Row(elem_classes="input-row"):
        base = gr.ImageEditor(
            label="Background Image (Normal Object)",
            type="pil",
            width=420,
            height=450,
            canvas_size=(CANVAS_W, CANVAS_H),
            sources=["upload"],
            brush=gr.Brush(
                default_size=15,
                default_color="#FFFFFF",
                color_mode="fixed",
                colors=["#FFFFFF"],
            ),
        )
        ref = gr.Image(
            label="Reference Image (Reference Anomaly)",
            sources=["upload"],
            type="pil",
            width=420,
            height=380,
        )

    with gr.Row():
        gen_btn = gr.Button("Generate", variant="primary")

    # ==================== Generation Result + Attention Map Side-by-Side ====================
    with gr.Row():
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Result",
                interactive=False,
            )
        with gr.Column(scale=1):
            attention_map_output = gr.Image(
                label="Attention Map (Attention Visualization for Reference Anomaly)",
                interactive=False,
            )

    with gr.Row():
        error_dialog = gr.HTML(visible=False)

    gr.HTML('<div class="ex-section-header"><span style="font-weight:700;font-size:16px;color:#374151;">Examples (click to load background + mask + reference image)</span></div>')

    ex_btn0 = gr.Button("Example 1", visible=False, elem_id="ex_btn_0")
    ex_btn1 = gr.Button("Example 2", visible=False, elem_id="ex_btn_1")
    ex_btn2 = gr.Button("Example 3", visible=False, elem_id="ex_btn_2")
    ex_btn3 = gr.Button("Example 4", visible=False, elem_id="ex_btn_3")

    gr.HTML('<div class="ex-container">' + build_examples_html() + '</div>')

    # ==================== Event Bindings ====================

    ex_btn0.click(fn=load_ex1, outputs=[base, ref])
    ex_btn1.click(fn=load_ex2, outputs=[base, ref])
    ex_btn2.click(fn=load_ex3, outputs=[base, ref])
    ex_btn3.click(fn=load_ex4, outputs=[base, ref])

    gen_btn.click(
        fn=run_local,
        inputs=[base, ref],
        outputs=[output_image, attention_map_output, error_dialog],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
