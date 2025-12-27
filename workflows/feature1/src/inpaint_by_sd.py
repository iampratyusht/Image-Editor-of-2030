# inpaint_by_sd.py
# =================
# Stable Diffusion Inpainting Module with LCM Scheduler
#
# Functions:
#   - build_sd_inpaint_model(): Build SD model with auto-download support
#   - inpaint_with_sd(): Simple inpainting function
#   - replace_img_with_sd(): Replace segmented object with new content (8 steps with LCM)
#   - fill_img_with_sd(): Fill segmented region with new content (8 steps with LCM)
#   - fill_background_with_sd(): Fill background while keeping object
#
# Uses Stable Diffusion Inpainting with LCM-LoRA for fast inference.
# Mask convention: WHITE (255) = area to replace/fill with generated content

import os
import warnings
import cv2
import torch
import numpy as np
import PIL.Image as Image
from PIL import Image as PILImage
from typing import Union, Optional
from diffusers import (
    AutoPipelineForInpainting,
    LCMScheduler,
)
from ..utils.crop_for_replacing import recover_size, resize_and_pad
from ..utils.mask_processing import crop_for_filling_pre, crop_for_filling_post

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# --- Global cache for SD pipeline (load once) ---
_sd_pipe = None
_sd_fill_pipe = None

# Model paths
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
SD_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "stable-diffusion-inpainting")
SD_FILL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "stable-diffusion-inpainting")
LCM_LORA_PATH = os.path.join(CHECKPOINT_DIR, "lcm-lora-sdv1-5")


def download_sd_model_to_checkpoints():
    """Download SD inpainting model to local checkpoints folder."""
    from huggingface_hub import snapshot_download
    
    if not os.path.exists(SD_MODEL_PATH):
        print(f"  Downloading SD Inpainting model to {SD_MODEL_PATH}...")
        snapshot_download(
            repo_id="runwayml/stable-diffusion-inpainting",
            local_dir=SD_MODEL_PATH,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"  SD model found at {SD_MODEL_PATH}")
    
    return SD_MODEL_PATH


def download_lcm_lora_to_checkpoints():
    """Download LCM-LoRA weights to local checkpoints folder."""
    from huggingface_hub import snapshot_download
    
    if not os.path.exists(LCM_LORA_PATH):
        print(f"  Downloading LCM-LoRA to {LCM_LORA_PATH}...")
        snapshot_download(
            repo_id="latent-consistency/lcm-lora-sdv1-5",
            local_dir=LCM_LORA_PATH,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"  LCM-LoRA found at {LCM_LORA_PATH}")
    
    return LCM_LORA_PATH


def download_lcm_lora_to_checkpoints():
    """Download LCM-LoRA weights to local checkpoints folder."""
    from huggingface_hub import snapshot_download
    
    if not os.path.exists(LCM_LORA_PATH):
        print(f"  Downloading LCM-LoRA weights to {LCM_LORA_PATH}...")
        snapshot_download(
            repo_id="latent-consistency/lcm-lora-sdv1-5",
            local_dir=LCM_LORA_PATH,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"  LCM-LoRA found at {LCM_LORA_PATH}")
    
    return LCM_LORA_PATH


def build_sd_inpaint_model(device: str = "cpu", use_lcm: bool = True, download_to_checkpoints: bool = True):
    """
    Build Stable Diffusion inpainting model with LCM-LoRA for fast inference.
    
    Args:
        device: "cpu" or "cuda"
        use_lcm: Whether to use LCM-LoRA for faster inference (8 steps instead of 50)
        download_to_checkpoints: If True, download models to ./checkpoints instead of HF cache
    
    Returns:
        Configured pipeline ready for inpainting
    """
    # Determine model path
    if download_to_checkpoints:
        sd_model_path = download_sd_model_to_checkpoints()
    else:
        sd_model_path = "runwayml/stable-diffusion-inpainting"
    
    print(f"  Loading Stable Diffusion Inpainting model from {sd_model_path}...")
    
    # Load base inpainting model
    pipe = AutoPipelineForInpainting.from_pretrained(
        sd_model_path,
        torch_dtype=torch.float32,
        local_files_only=download_to_checkpoints,
    ).to(device)
    
    if use_lcm:
        print("  Setting up LCM scheduler for fast inference...")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        # Determine LCM-LoRA path
        if download_to_checkpoints:
            lcm_lora_path = download_lcm_lora_to_checkpoints()
        else:
            lcm_lora_path = "latent-consistency/lcm-lora-sdv1-5"
        
        # Load and fuse LCM-LoRA weights
        print(f"  Loading LCM-LoRA weights from {lcm_lora_path}...")
        pipe.load_lora_weights(lcm_lora_path)
        pipe.fuse_lora()
    
    print("  SD Inpainting model ready!")
    return pipe


def inpaint_with_sd(
    pipe,
    image: Union[np.ndarray, PILImage.Image],
    mask: Union[np.ndarray, PILImage.Image],
    prompt: str = "high quality, detailed, realistic background",
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 8,
    guidance_scale: float = 4.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Inpaint an image using Stable Diffusion with LCM.
    
    Args:
        pipe: The SD inpainting pipeline
        image: Input image (RGB, numpy array H,W,3 or PIL Image)
        mask: Binary mask where white (255) = area to inpaint
        prompt: Text prompt describing what to generate in masked area
        negative_prompt: What to avoid in generation
        num_inference_steps: Number of denoising steps (8 for LCM)
        guidance_scale: How closely to follow the prompt (4.0 for LCM)
        seed: Random seed for reproducibility
    
    Returns:
        Inpainted image as numpy array (RGB, H,W,3)
    """
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        init_image = PILImage.fromarray(image)
    else:
        init_image = image
    
    if isinstance(mask, np.ndarray):
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_image = PILImage.fromarray(mask)
    else:
        mask_image = mask
    
    orig_size = init_image.size  # (W, H)
    target_size = 512
    
    # Resize maintaining aspect ratio
    w, h = orig_size
    if max(w, h) > target_size:
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        init_image = init_image.resize((new_w, new_h), PILImage.LANCZOS)
        mask_image = mask_image.resize((new_w, new_h), PILImage.NEAREST)
    else:
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            init_image = init_image.resize((new_w, new_h), PILImage.LANCZOS)
            mask_image = mask_image.resize((new_w, new_h), PILImage.NEAREST)
    
    generator = torch.manual_seed(seed) if seed is not None else None
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    if result.size != orig_size:
        result = result.resize(orig_size, PILImage.LANCZOS)
    
    return np.array(result)

def _make_pipe(device: str):
    """Create SD Inpainting pipeline with LCM scheduler from local checkpoints."""
    global _sd_pipe
    
    # Return cached pipe if already loaded
    if _sd_pipe is not None:
        return _sd_pipe
    
    # Ensure model is downloaded
    if not os.path.exists(SD_MODEL_PATH):
        download_sd_model_to_checkpoints()
    
    # Ensure LCM-LoRA is downloaded
    if not os.path.exists(LCM_LORA_PATH):
        download_lcm_lora_to_checkpoints()
    
    print("  Loading SD Inpainting from checkpoints (float32)...")
    
    # Load pipeline from local checkpoint with float32 for CPU compatibility
    _sd_pipe = AutoPipelineForInpainting.from_pretrained(
        SD_MODEL_PATH,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    
    # Set LCM scheduler for fast inference
    _sd_pipe.scheduler = LCMScheduler.from_config(_sd_pipe.scheduler.config)
    
    # Load and fuse LCM-LoRA weights
    print(f"  Loading LCM-LoRA from {LCM_LORA_PATH}...")
    _sd_pipe.load_lora_weights(LCM_LORA_PATH)
    _sd_pipe.fuse_lora()
    
    _sd_pipe = _sd_pipe.to(device)
    
    print("  SD Inpainting with LCM ready!")
    return _sd_pipe

def _preprocess_mask(mask):
    if mask.ndim == 3: mask = mask[..., 0]
    return (mask > 127).astype(np.uint8) * 255

def replace_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, deformed, ugly",
        step: int = 8,
        device="cpu"
):
    """
    Replace the segmented/masked region with new content using SD Inpainting with LCM.
    
    The MASKED AREA (white pixels, 255) will be replaced with generated content.
    Uses resize_and_pad -> inpaint -> recover_size -> composite.
    
    Args:
        img: Input image (RGB numpy array)
        mask: Binary mask (255 = area to REPLACE with new content)
        text_prompt: Description of what to generate in the masked area
        negative_prompt: What to avoid in generation
        step: Number of inference steps (default: 8 for LCM)
        device: Device to run on
        
    Returns:
        Image with masked region replaced (RGB numpy array)
    """
    pipe = _make_pipe(device)

    # Preprocess mask
    mask_proc = _preprocess_mask(mask)

    
    # Resize and pad for SD processing
    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask_proc)
    
    # Run inpainting
    img_padded_result = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_padded),
        mask_image=Image.fromarray(mask_padded),
        num_inference_steps=step,
        guidance_scale=4,
    ).images[0]
    
    # Recover original size
    height, width, _ = img.shape
    img_resized, mask_resized = recover_size(
        np.array(img_padded_result), mask_padded, (height, width), padding_factors)
    
    # Composite: use generated content in masked area, original elsewhere
    mask_blend = np.expand_dims(mask_resized, -1) / 255
    img_replaced = img_resized * mask_blend + img * (1 - mask_blend)
    
    return img_replaced.astype(np.uint8)


def _make_fill_pipe(device: str):
    """Create SD Inpainting pipeline with LCM scheduler for fill_img_with_sd()."""
    global _sd_fill_pipe
    
    # Return cached pipe if already loaded
    if _sd_fill_pipe is not None:
        return _sd_fill_pipe
    
    dtype = torch.float32  # Use float32 for CPU inference
    
    print("  Loading SD Inpainting with LCM scheduler for fill...")
    
    # Load SD Inpainting pipeline from local checkpoint
    _sd_fill_pipe = AutoPipelineForInpainting.from_pretrained(
        SD_FILL_MODEL_PATH,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    
    # Set LCM scheduler for fast inference
    _sd_fill_pipe.scheduler = LCMScheduler.from_config(_sd_fill_pipe.scheduler.config)
    
    # Load and fuse LCM-LoRA weights
    if os.path.exists(LCM_LORA_PATH):
        print(f"  Loading LCM-LoRA from {LCM_LORA_PATH}...")
        _sd_fill_pipe.load_lora_weights(LCM_LORA_PATH)
        _sd_fill_pipe.fuse_lora()
    
    _sd_fill_pipe = _sd_fill_pipe.to(device)
    
    # Enable memory optimizations for CPU
    try:
        _sd_fill_pipe.enable_attention_slicing()
    except:
        pass
    
    print("  SD Inpainting with LCM for fill ready!")
    return _sd_fill_pipe


def fill_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        num_inference_steps: int = 8,
        guidance_scale: float = 4.0,
        device="cpu"
):
    """
    Fill the segmented/masked region with new content using SD Inpainting with LCM.
    
    Uses smart cropping around the mask for better context-aware filling.
    The MASKED AREA (white pixels, 255) will be filled with generated content.
    Uses LCM scheduler for fast inference (8 steps).
    
    Args:
        img: Input image (RGB numpy array)
        mask: Binary mask (255 = area to FILL with new content)
        text_prompt: Description of what to generate in the masked area
        num_inference_steps: Number of diffusion steps (default: 8 for LCM)
        guidance_scale: Guidance scale for generation (default: 4.0 for LCM)
        device: Device to run on
        
    Returns:
        Image with masked region filled (RGB numpy array)
    """
    # Use SD Inpainting with LCM pipeline for filling
    pipe = _make_fill_pipe(device)
    
    # Preprocess mask
    mask_proc = _preprocess_mask(mask)
    
    # Crop around the mask for better context-aware filling
    img_crop, mask_crop = crop_for_filling_pre(img, mask_proc)
    
    # Run SD Inpainting with LCM on cropped region
    # mask indicates area to fill (standard SD inpainting behavior)
    result = pipe(
        prompt=text_prompt,
        negative_prompt="blurry, low quality, distorted, deformed",
        image=Image.fromarray(img_crop),
        mask_image=Image.fromarray(mask_crop),  # White = area to fill
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    # Post-process: paste filled region back into original image
    img_filled = crop_for_filling_post(img, mask_proc, np.array(result))
    
    return img_filled


def fill_background_with_sd(img: np.ndarray, mask: np.ndarray, text_prompt: str,
                            step: int = 8, device: str = "cpu"):
    """
    Fill the BACKGROUND with stable diffusion while keeping the masked object.
    
    Args:
        img: Original image (H, W, 3) BGR or RGB
        mask: Object mask (H, W) - WHITE (255) = object to KEEP
        text_prompt: Description of the new background
        step: Number of inference steps
        device: Device to use
    
    Returns:
        Image with new background, original object preserved
    
    Note:
        This function INVERTS the mask so that the background (unmasked area)
        becomes the area to inpaint, while the object is preserved.
    """
    pipe = _make_pipe(device)
    
    # Preprocess mask (ensures 0/255 values)
    mask_proc = _preprocess_mask(mask)
    
    # INVERT the mask: object (WHITE) -> background (WHITE)
    # So now WHITE = background = area to fill
    inverted_mask = 255 - mask_proc
    
    # Resize to SD-compatible dimensions
    height, width = img.shape[:2]
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    
    img_resized = cv2.resize(img, (new_width, new_height))
    mask_resized = cv2.resize(inverted_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Ensure mask is 2D
    if len(mask_resized.shape) == 3:
        mask_resized = mask_resized[:, :, 0]
    
    # Run SD inpainting on the BACKGROUND (inverted mask area)
    result = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_resized),
        mask_image=Image.fromarray(mask_resized),
        num_inference_steps=step,
        guidance_scale=4,
    ).images[0]
    
    result_np = np.array(result)
    
    # Resize result back to original dimensions
    result_np = cv2.resize(result_np, (width, height))
    
    # Composite: keep original object, use generated background
    # mask_proc: WHITE (255) = object to keep
    mask_float = mask_proc.astype(np.float32) / 255.0
    if len(mask_float.shape) == 2:
        mask_float = mask_float[:, :, np.newaxis]
    
    # object_area = original * mask + generated * (1 - mask)
    # Where mask=1 (object) -> keep original, mask=0 (background) -> use generated
    final = (img.astype(np.float32) * mask_float + 
             result_np.astype(np.float32) * (1.0 - mask_float))
    
    return final.astype(np.uint8)