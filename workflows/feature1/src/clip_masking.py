"""
CLIP-based Object Segmentation

Uses CLIP for text-image matching combined with SAM automatic mask generation
to find objects by text description.

Usage:
    from clip_segment import find_object_with_text, CLIP_AVAILABLE
    
    if CLIP_AVAILABLE:
        mask = find_object_with_text(img, "red car on the left", sam_ckpt, device)
"""

import numpy as np
import torch
import cv2
from typing import List, Optional, Dict, Any
from PIL import Image

# Try to import CLIP
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Note: Install open-clip-torch for text prompt mode: pip install open-clip-torch")

# Use mobilesamsegment instead of segment_anything
try:
    from .mobilesamsegment import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Note: mobilesamsegment not available for automatic mask generation")


# --- Global cache for models (lazy loading) ---
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_mask_generator = None
_mask_generator_ckpt = None  # Track which checkpoint is loaded

# Local CLIP checkpoint path (set this if you have CLIP downloaded locally)
CLIP_LOCAL_PATH = "./feature1/checkpoints/clip-vit-b-32"


def download_clip_to_checkpoints():
    """Download CLIP model to local checkpoints folder."""
    import os
    from huggingface_hub import hf_hub_download
    
    os.makedirs(CLIP_LOCAL_PATH, exist_ok=True)
    local_model_path = os.path.join(CLIP_LOCAL_PATH, "open_clip_pytorch_model.bin")
    
    # Check if file exists and is valid (at least 500MB for ViT-B-32)
    min_size = 500 * 1024 * 1024  # 500 MB
    needs_download = False
    
    if not os.path.exists(local_model_path):
        needs_download = True
    elif os.path.getsize(local_model_path) < min_size:
        print(f" -> CLIP model file appears corrupted (too small), re-downloading...")
        os.remove(local_model_path)
        needs_download = True
    
    if needs_download:
        print(f" -> Downloading CLIP model to {CLIP_LOCAL_PATH}...")
        hf_hub_download(
            repo_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            filename="open_clip_pytorch_model.bin",
            local_dir=CLIP_LOCAL_PATH,
            local_dir_use_symlinks=False,
        )
        print(" -> CLIP model downloaded.")
        
        # Verify download
        if os.path.exists(local_model_path) and os.path.getsize(local_model_path) >= min_size:
            print(f" -> CLIP model verified ({os.path.getsize(local_model_path) / 1024 / 1024:.1f} MB)")
        else:
            raise RuntimeError("CLIP model download verification failed - file too small or missing")
    else:
        print(f" -> CLIP model found at {CLIP_LOCAL_PATH}")
    
    return local_model_path


def get_clip_model(device: str = "cpu"):
    """
    Lazy load CLIP model (ViT-B-32).
    Downloads to checkpoints folder if not present.
    
    Returns:
        tuple: (clip_model, preprocess, tokenizer)
    """
    global _clip_model, _clip_preprocess, _clip_tokenizer
    
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP not available. Install with: pip install open-clip-torch")
    
    if _clip_model is None:
        import os
        
        # Check for local checkpoint, download if not present
        local_model_path = os.path.join(CLIP_LOCAL_PATH, "open_clip_pytorch_model.bin")
        
        # Always verify/download through the download function
        try:
            download_clip_to_checkpoints()
        except Exception as e:
            print(f"\n    ERROR: Failed to download CLIP model!")
            print(f"    Reason: {type(e).__name__}: {e}")
            print(f"\n    SOLUTIONS:")
            print(f"    1. Check your internet connection")
            print(f"    2. Download manually and place in: {CLIP_LOCAL_PATH}/")
            print(f"       Download URL: https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
            print(f"    3. Use Box/Point/Sketch mode instead of Text mode\n")
            raise
        
        print(f" -> Loading CLIP model from: {CLIP_LOCAL_PATH}")
        try:
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=local_model_path
            )
            
            _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _clip_model.eval().to(device)
            print(" -> CLIP model loaded.")
        except Exception as load_err:
            # If loading fails, the file might be corrupted - delete and retry once
            print(f" -> CLIP model loading failed: {load_err}")
            print(" -> Attempting to re-download...")
            if os.path.exists(local_model_path):
                os.remove(local_model_path)
            download_clip_to_checkpoints()
            
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=local_model_path
            )
            _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _clip_model.eval().to(device)
            print(" -> CLIP model loaded after re-download.")
    
    return _clip_model, _clip_preprocess, _clip_tokenizer


def get_mask_generator(
        sam_ckpt: str, 
        model_type: str = "vit_t", 
        device: str = "cpu",
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100
):
    """
    Lazy load SAM Automatic Mask Generator.
    
    Args:
        sam_ckpt: Path to SAM checkpoint
        model_type: SAM model type ('vit_t', 'vit_b', 'vit_l', 'vit_h')
        device: Device to run on
        points_per_side: Number of points per side for mask generation grid
        pred_iou_thresh: IoU threshold for predictions
        stability_score_thresh: Stability score threshold
        min_mask_region_area: Minimum mask region area
    
    Returns:
        SamAutomaticMaskGenerator instance
    """
    global _mask_generator, _mask_generator_ckpt
    
    if not SAM_AVAILABLE:
        raise RuntimeError("SAM not available for automatic mask generation")
    
    # Reload if checkpoint changed
    if _mask_generator is None or _mask_generator_ckpt != sam_ckpt:
        print(" -> Loading SAM Automatic Mask Generator...")
        sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
        sam.to(device)
        sam.eval()
        
        _mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area
        )
        _mask_generator_ckpt = sam_ckpt
        print(" -> SAM Mask Generator loaded.")
    
    return _mask_generator


def get_spatial_weight(bbox: List[int], img_w: int, img_h: int, prompt: str) -> float:
    """
    Calculate spatial penalty based on prompt keywords.
    
    Supports keywords: left, right, center, top, bottom
    
    Args:
        bbox: Bounding box [x, y, w, h]
        img_w: Image width
        img_h: Image height
        prompt: Text prompt that may contain spatial keywords
    
    Returns:
        Spatial weight (0.0 to 1.0), lower means less likely match
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    
    prompt_lower = prompt.lower()
    score = 1.0
    
    # Horizontal position
    if "left" in prompt_lower and cx > 0.6:
        score *= 0.1
    if "right" in prompt_lower and cx < 0.4:
        score *= 0.1
    if "center" in prompt_lower and (cx < 0.3 or cx > 0.7):
        score *= 0.5
    
    # Vertical position
    if "top" in prompt_lower and cy > 0.6:
        score *= 0.3
    if "bottom" in prompt_lower and cy < 0.4:
        score *= 0.3
    
    return score


def generate_all_masks(
        img: np.ndarray,
        sam_ckpt: str,
        model_type: str = "vit_t",
        device: str = "cpu",
        resize_target: int = 640
) -> List[Dict[str, Any]]:
    """
    Generate all masks in an image using SAM Automatic Mask Generator.
    
    Args:
        img: Input image (RGB numpy array)
        sam_ckpt: Path to SAM checkpoint
        model_type: SAM model type ('vit_t', 'vit_b', 'vit_l', 'vit_h')
        device: Device to run on
        resize_target: Resize longest side to this for faster processing
    
    Returns:
        List of mask dictionaries with 'segmentation', 'bbox', 'area', etc.
    """
    h, w = img.shape[:2]
    
    # Resize for faster processing
    scale = resize_target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Get mask generator
    mask_gen = get_mask_generator(sam_ckpt, model_type, device)
    
    print(f" -> Generating masks ({new_w}x{new_h})...")
    masks = mask_gen.generate(img_resized)
    
    # Add scale info for upscaling later
    for m in masks:
        m['_scale'] = scale
        m['_original_size'] = (h, w)
        m['_resized_size'] = (new_h, new_w)
    
    return masks


def score_masks_with_clip(
        img_resized: np.ndarray,
        masks: List[Dict[str, Any]],
        text_prompt: str,
        device: str = "cpu"
) -> List[float]:
    """
    Score masks using CLIP text-image similarity.
    
    Args:
        img_resized: Resized image (RGB numpy array)
        masks: List of mask dictionaries from SAM
        text_prompt: Text description to match
        device: Device to run on
    
    Returns:
        List of similarity scores for each mask
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP not available")
    
    # Load CLIP
    clip_m, preprocess, tokenizer = get_clip_model(device)
    
    # Prepare crops for CLIP
    crops = []
    for m in masks:
        x, y, wb, hb = map(int, m['bbox'])
        crop = img_resized[y:y+hb, x:x+wb].copy()
        
        # Mask out background (black) so CLIP sees only the object
        mask_crop = m['segmentation'][y:y+hb, x:x+wb]
        crop[~mask_crop] = 0
        
        # Convert to PIL and preprocess
        crop_pil = Image.fromarray(crop)
        crops.append(preprocess(crop_pil))
    
    # Batch encode with CLIP
    image_batch = torch.stack(crops).to(device)
    text_tokens = tokenizer([f"a photo of {text_prompt}"]).to(device)
    
    with torch.no_grad():
        img_emb = clip_m.encode_image(image_batch)
        txt_emb = clip_m.encode_text(text_tokens)
        
        # Normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        
        # Similarity scores
        sim_scores = (100.0 * img_emb @ txt_emb.T).softmax(dim=0).cpu().numpy().flatten()
    
    return sim_scores.tolist()


def find_object_with_text(
        img: np.ndarray,
        text_prompt: str,
        sam_ckpt: str,
        model_type: str = "vit_t",
        device: str = "cpu",
        min_area_ratio: float = 0.005,
        size_bonus: float = 0.25
) -> Optional[np.ndarray]:
    """
    Find object in image using CLIP + SAM automatic mask generation.
    
    This function:
    1. Generates all possible masks using SAM automatic mask generator
    2. Filters masks by size and spatial location (based on prompt keywords)
    3. Scores each mask using CLIP text-image similarity
    4. Returns the best matching mask
    
    Args:
        img: Input image (RGB numpy array)
        text_prompt: Text description of object to find
                     Supports spatial keywords: "left", "right", "center", "top", "bottom"
                     Example: "red car on the left", "person in the center"
        sam_ckpt: Path to SAM checkpoint
        model_type: SAM model type ('vit_t', 'vit_b', 'vit_l', 'vit_h')
        device: Device to run on ('cpu')
        min_area_ratio: Minimum mask area as ratio of image (default 0.5%)
        size_bonus: Bonus weight for larger objects (default 0.25)
    
    Returns:
        Binary mask as uint8 (0 or 255) at original resolution, or None if not found
    """
    if not CLIP_AVAILABLE:
        print("Error: CLIP not available. Install with: pip install open-clip-torch")
        return None
    
    if not SAM_AVAILABLE:
        print("Error: SAM not available for automatic mask generation")
        return None
    
    h, w = img.shape[:2]
    
    # Generate all masks
    masks = generate_all_masks(img, sam_ckpt, model_type, device)
    
    if len(masks) == 0:
        print(" -> No masks generated.")
        return None
    
    # Get resize info
    scale = masks[0]['_scale']
    new_h, new_w = masks[0]['_resized_size']
    img_resized = cv2.resize(img, (new_w, new_h))
    total_area = new_w * new_h
    
    # Filter masks by size and spatial location
    valid_masks = []
    spatial_weights = []
    
    for m in masks:
        # Filter tiny masks
        if m['area'] < (total_area * min_area_ratio):
            continue
        
        # Calculate spatial score
        sp_score = get_spatial_weight(m['bbox'], new_w, new_h, text_prompt)
        
        # Skip if definitely wrong location
        if sp_score < 0.15:
            continue
        
        valid_masks.append(m)
        spatial_weights.append(sp_score)
    
    if len(valid_masks) == 0:
        print(" -> No valid masks after filtering.")
        return None
    
    print(f" -> Scoring {len(valid_masks)} candidate masks with CLIP...")
    
    # Score with CLIP
    sim_scores = score_masks_with_clip(img_resized, valid_masks, text_prompt, device)
    
    # Final scoring: CLIP similarity * spatial weight + size bonus
    final_scores = []
    for i, sim in enumerate(sim_scores):
        area_ratio = valid_masks[i]['area'] / total_area
        # Score = (visual match * location match) + size bonus
        f_score = (sim * spatial_weights[i]) + (area_ratio * size_bonus)
        final_scores.append(f_score)
    
    # Select best
    best_idx = int(np.argmax(final_scores))
    best_mask = valid_masks[best_idx]
    
    print(f" -> Best match: score={final_scores[best_idx]:.3f}, area={best_mask['area']}")
    
    # Upscale mask to original resolution
    full_mask = cv2.resize(
        best_mask['segmentation'].astype(np.uint8) * 255,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )
    
    return full_mask


def find_multiple_objects_with_text(
        img: np.ndarray,
        text_prompt: str,
        sam_ckpt: str,
        model_type: str = "vit_t",
        device: str = "cpu",
        top_k: int = 3,
        score_threshold: float = 0.1
) -> List[np.ndarray]:
    """
    Find multiple objects matching the text prompt.
    
    Args:
        img: Input image (RGB numpy array)
        text_prompt: Text description of objects to find
        sam_ckpt: Path to SAM checkpoint
        model_type: SAM model type
        device: Device to run on
        top_k: Maximum number of masks to return
        score_threshold: Minimum score threshold
    
    Returns:
        List of binary masks as uint8 (0 or 255) at original resolution
    """
    if not CLIP_AVAILABLE or not SAM_AVAILABLE:
        return []
    
    h, w = img.shape[:2]
    
    # Generate all masks
    masks = generate_all_masks(img, sam_ckpt, model_type, device)
    
    if len(masks) == 0:
        return []
    
    # Get resize info
    scale = masks[0]['_scale']
    new_h, new_w = masks[0]['_resized_size']
    img_resized = cv2.resize(img, (new_w, new_h))
    total_area = new_w * new_h
    
    # Filter masks
    valid_masks = []
    spatial_weights = []
    
    for m in masks:
        if m['area'] < (total_area * 0.005):
            continue
        sp_score = get_spatial_weight(m['bbox'], new_w, new_h, text_prompt)
        if sp_score < 0.15:
            continue
        valid_masks.append(m)
        spatial_weights.append(sp_score)
    
    if len(valid_masks) == 0:
        return []
    
    # Score with CLIP
    sim_scores = score_masks_with_clip(img_resized, valid_masks, text_prompt, device)
    
    # Calculate final scores
    final_scores = []
    for i, sim in enumerate(sim_scores):
        area_ratio = valid_masks[i]['area'] / total_area
        f_score = (sim * spatial_weights[i]) + (area_ratio * 0.25)
        final_scores.append((f_score, i))
    
    # Sort by score and get top_k
    final_scores.sort(reverse=True)
    
    result_masks = []
    for score, idx in final_scores[:top_k]:
        if score < score_threshold:
            break
        mask = valid_masks[idx]
        full_mask = cv2.resize(
            mask['segmentation'].astype(np.uint8) * 255,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        result_masks.append(full_mask)
    
    return result_masks
