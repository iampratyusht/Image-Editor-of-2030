"""
LaMa Inpainting Module
Supports multiple LaMa variants: big-lama, lama-dilated, lama-fourier, qualcomm
"""

import os
import sys
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lama"))

from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.data import pad_tensor_to_modulo

from ..utils import load_img_to_array, save_array_to_img


@torch.no_grad()
def inpaint_img_with_lama(img, mask, config_p, ckpt_p, mod=8, device="cpu"):
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(predict_config.model.path, 'models',
                                   predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

    batch = {
        'image': img.permute(2, 0, 1).unsqueeze(0),
        'mask': mask[None, None]
    }
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)

    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0).float()

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = cur_res[:unpad_to_size[0], :unpad_to_size[1]]
    return np.clip(cur_res * 255, 0, 255).astype('uint8')


def build_lama_model(config_p, ckpt_p, device="cpu"):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(predict_config.model.path, 'models',
                                   predict_config.model.checkpoint)

    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    model.freeze()
    return model


@torch.no_grad()
def inpaint_img_with_builded_lama(model, img, mask, config_p=None, mod=8, device="cpu"):
    if np.max(mask) == 1:
        mask = mask * 255
    
    # Ensure image is in correct format
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)  # Convert grayscale to RGB
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # Remove alpha channel if present
    
    # Ensure mask is 2D grayscale
    if mask.ndim == 3:
        mask = mask[:, :, 0]  # Take first channel if multi-channel
    
    # Normalize to [0, 1]
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float().div(255.)
    
    # Reshape for model input: (C, H, W) then add batch dimension
    batch = {
        'image': img.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H, W)
        'mask': mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    }
    
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)

    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0).float()

    batch = model(batch)
    cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = cur_res[:unpad_to_size[0], :unpad_to_size[1]]
    return np.clip(cur_res * 255, 0, 255).astype('uint8')


# def main():
#     import argparse
#     import time
    
#     parser = argparse.ArgumentParser(description="LaMa Inpainting - Standalone Test")
#     parser.add_argument("--input_img", type=str, help="Path to input image")
#     parser.add_argument("--input_mask", type=str, help="Path to mask image (white=inpaint region)")
#     parser.add_argument("--output_dir", type=str, default="./results/lama_test", help="Output directory")
#     parser.add_argument("--lama_config", type=str, default="./lama/configs/prediction/default.yaml", help="LaMa config path")
#     parser.add_argument("--lama_ckpt", type=str, default="./pretrained_models/lama-dilated", help="LaMa checkpoint directory")
#     parser.add_argument("--dummy", action="store_true", help="Use dummy input (512x512) for testing - same as ONNX export script")
#     parser.add_argument("--dummy_size", type=int, default=512, help="Size of dummy input (default: 512)")
#     args = parser.parse_args()
    
#     # Validate args
#     if not args.dummy and (not args.input_img or not args.input_mask):
#         parser.error("Either --dummy flag OR both --input_img and --input_mask are required")
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Setup device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"\n{'='*50}")
#     print(f"  LaMa Inpainting Test")
#     print(f"{'='*50}")
#     print(f"  Device: {device}")
    
#     if args.dummy:
#         # Create dummy inputs matching ONNX export script
#         print(f"  Mode: DUMMY INPUT TEST")
#         print(f"  Size: {args.dummy_size}x{args.dummy_size}")
#         print(f"  Checkpoint: {args.lama_ckpt}")
#         print(f"{'='*50}\n")
        
#         # Same dummy input as onnx_export.py
#         # Image: normalized RGB in [0, 1] range -> convert to [0, 255] uint8
#         # Mask: binary mask where 1 = area to inpaint
#         print("Creating dummy inputs (same as ONNX export)...")
#         np.random.seed(42)  # For reproducibility
#         img = (np.random.rand(args.dummy_size, args.dummy_size, 3) * 255).astype(np.uint8)
#         mask = np.zeros((args.dummy_size, args.dummy_size), dtype=np.uint8)
#         # Create rectangular region to inpaint (100:200, 100:200)
#         mask[100:200, 100:200] = 255
        
#         print(f"  Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
#         print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}, range: [{mask.min()}, {mask.max()}]")
#         input_name = "dummy"
#     else:
#         print(f"  Input: {args.input_img}")
#         print(f"  Mask: {args.input_mask}")
#         print(f"  Checkpoint: {args.lama_ckpt}")
#         print(f"{'='*50}\n")
        
#         # Load image and mask
#         print("Loading image and mask...")
#         img = load_img_to_array(args.input_img)
#         mask = load_img_to_array(args.input_mask)
        
#         # Convert mask to grayscale if needed
#         if mask.ndim == 3:
#             import cv2
#             mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
#         print(f"  Image shape: {img.shape}")
#         print(f"  Mask shape: {mask.shape}")
#         print(f"  Mask range: [{mask.min()}, {mask.max()}]")
#         input_name = Path(args.input_img).stem
    
#     # Build model
#     print("\nLoading LaMa model...")
#     model_load_start = time.time()
#     model = build_lama_model(args.lama_config, args.lama_ckpt, device=device)
#     model_load_time = time.time() - model_load_start
#     print(f"  Model loaded in {model_load_time:.2f}s")
    
#     # Inpaint
#     print("\nRunning inpainting...")
#     inpaint_start = time.time()
#     result = inpaint_img_with_builded_lama(model, img, mask, args.lama_config, device=device)
#     inpaint_time = time.time() - inpaint_start
    
#     # Save results
#     result_path = os.path.join(args.output_dir, f"{input_name}_inpainted.png")
#     save_array_to_img(result, result_path)
    
#     # Also save a side-by-side comparison
#     import cv2
#     h, w = img.shape[:2]
#     comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
#     comparison[:, :w] = img
#     mask_rgb = np.stack([mask, mask, mask], axis=2) if mask.ndim == 2 else mask
#     comparison[:, w:w*2] = mask_rgb
#     comparison[:, w*2:] = result
#     comparison_path = os.path.join(args.output_dir, f"{input_name}_comparison.png")
#     save_array_to_img(comparison, comparison_path)
    
#     # Print summary
#     print(f"\n{'='*50}")
#     print(f"  RESULTS")
#     print(f"{'='*50}")
#     print(f"  Model Load Time:  {model_load_time:.3f}s")
#     print(f"  Inpaint Time:     {inpaint_time:.3f}s")
#     print(f"  Total Time:       {model_load_time + inpaint_time:.3f}s")
#     print(f"{'='*50}")
#     print(f"\n  Saved to:")
#     print(f"    Result: {result_path}")
#     print(f"    Comparison: {comparison_path}")
#     print(f"{'='*50}\n")


# if __name__ == "__main__":
#     main()

