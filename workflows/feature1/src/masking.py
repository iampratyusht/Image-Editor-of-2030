"""
SAM Segmentation Functions

Provides functions for mask prediction using SAM models (SAM, MobileSAM).

Usage:
    from sam_segment import predict_masks_with_sam, build_sam_model
    
    masks, scores, logits = predict_masks_with_sam(
        img, point_coords=[[x, y]], point_labels=[1],
        model_type="vit_t", ckpt_p="pretrained_models/mobile_sam.pt"
    )
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List, Optional
import torch

# Use mobilesamsegment instead of segment_anything
from .mobilesamsegment import SamPredictor, sam_model_registry
from ..utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points

# --- Global cache for SAM model (load once) ---
_sam_model = None
_sam_predictor = None
_sam_model_type = None
_sam_ckpt = None


def get_available_models() -> List[str]:
    """Get list of available SAM models."""
    try:
        return list(sam_model_registry.keys())
    except:
        return ['vit_h', 'vit_l', 'vit_b', 'vit_t']


def get_sam_predictor(model_type: str, ckpt_p: str, device: str = "cpu"):
    """Get or create cached SAM predictor."""
    global _sam_model, _sam_predictor, _sam_model_type, _sam_ckpt
    
    # Return cached predictor if same model
    if _sam_predictor is not None and _sam_model_type == model_type and _sam_ckpt == ckpt_p:
        return _sam_predictor
    
    # Load new model
    _sam_model = sam_model_registry[model_type](checkpoint=ckpt_p)
    _sam_model.to(device=device)
    _sam_predictor = SamPredictor(_sam_model)
    _sam_model_type = model_type
    _sam_ckpt = ckpt_p
    
    return _sam_predictor


def predict_masks_with_sam(
        img: np.ndarray,
        point_coords: List[List[float]] = None,
        point_labels: List[int] = None,
        box_coords: List[float] = None,
        model_type: str = "vit_h",
        ckpt_p: str = None,
        device="cpu"
):
    # Get cached SAM predictor
    predictor = get_sam_predictor(model_type, ckpt_p, device)
    predictor.set_image(img)
    
    # Prepare prediction inputs
    pred_kwargs = {
        'multimask_output': True,
    }
    
    # Handle point prompts
    if point_coords is not None and len(point_coords) > 0:
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels) if point_labels is not None else np.ones(len(point_coords))
        pred_kwargs['point_coords'] = point_coords
        pred_kwargs['point_labels'] = point_labels
    
    # Handle box prompts
    if box_coords is not None and len(box_coords) > 0:
        box_coords = np.array(box_coords, dtype=np.float32).reshape(-1, 4)
        pred_kwargs['box'] = box_coords
    
    # Make prediction
    masks, scores, logits = predictor.predict(**pred_kwargs)
    return masks, scores, logits


def build_sam_model(model_type: str, ckpt_p: str, device="cpu"):
    
    return get_sam_predictor(model_type, ckpt_p, device)


# def setup_args(parser):
#     parser.add_argument("--input_img", type=str, required=True)
#     parser.add_argument("--point_coords", type=float, nargs='+', required=True)
#     parser.add_argument("--point_labels", type=int, nargs='+', required=True)
#     parser.add_argument("--dilate_kernel_size", type=int, default=None)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--sam_model_type", type=str, default="vit_h",
#                         choices=['vit_h', 'vit_t', 'vit_l', 'vit_b'])
#     parser.add_argument("--sam_ckpt", type=str, required=True)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     setup_args(parser)
#     args = parser.parse_args(sys.argv[1:])
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     img = load_img_to_array(args.input_img)

#     masks, _, _ = predict_masks_with_sam(
#         img,
#         [args.point_coords],
#         args.point_labels,
#         model_type=args.sam_model_type,
#         ckpt_p=args.sam_ckpt,
#         device=device,
#     )
#     masks = masks.astype(np.uint8) * 255


#     if args.dilate_kernel_size is not None:
#         masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

#     img_stem = Path(args.input_img).stem
#     out_dir = Path(args.output_dir) / img_stem
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for idx, mask in enumerate(masks):
#         mask_p = out_dir / f"mask_{idx}.png"
#         img_points_p = out_dir / f"with_points.png"
#         img_mask_p = out_dir / f"with_{Path(mask_p).name}"

#         save_array_to_img(mask, mask_p)

#         dpi = plt.rcParams['figure.dpi']
#         height, width = img.shape[:2]
#         plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
#         plt.imshow(img)
#         plt.axis('off')
#         show_points(plt.gca(), [args.point_coords], args.point_labels,
#                     size=(width*0.04)**2)
#         plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
#         show_mask(plt.gca(), mask, random_color=False)
#         plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
#         plt.close()
