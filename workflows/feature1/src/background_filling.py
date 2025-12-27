"""Interactive Background Filling Tool (Box, Point, Text, & Sketch)

This tool fills/replaces the BACKGROUND while keeping the selected object intact.
The user selects the OBJECT they want to keep, and the background is replaced.

Models:
  - Segmentation: MobileSAM (vit_t) - Fast and efficient
  - Inpainting: Stable Diffusion with LCM (8 steps)

Controls:
  [t] - Text Prompt (CLIP-based object detection)
  [b] - Box Selection
  [p] - Point Selection
  [s] - Sketch/Draw Selection
  [Enter] - Confirm selection (for Sketch/Points)
  [r] - Reset/Clear selection
  [q] - Quit

Usage:
  python background_filling.py --input_img <path> [--clip_prompt <text>] [--sd_prompt <text>]
"""

import cv2
import torch
import sys
import time
import numpy as np
import argparse
from pathlib import Path
import os

from .masking import predict_masks_with_sam
from .inpaint_by_sd import fill_background_with_sd
from ..utils import (
    load_img_to_array, save_array_to_img, dilate_mask,
    create_mask_overlay, create_comparison_grid, 
    save_all_results, save_minimal_results, print_saved_results
)

# Import CLIP-based segmentation
try:
    from .clip_masking import find_object_with_text, CLIP_AVAILABLE
except ImportError:
    CLIP_AVAILABLE = False
    print("Note: clip_masking.py not found or CLIP not installed. Text mode disabled.")

# --- GLOBAL VARIABLES ---
drawing = False
ix, iy = -1, -1
bbox = []
points = []
sketch_mask = None  # Holds the user's drawing

def fill_object(img, mask, pipe, device="cpu", dilate_kernel=15, fill_prompt="", step=50):

    # 2. Dilate Mask (protects object edges from background bleeding)
    final_mask = dilate_mask(mask, dilate_kernel)

    # 4. Fill BACKGROUND with Stable Diffusion (object preserved)
    print(f" -> Filling BACKGROUND with: '{fill_prompt}'...")
    print(f"    (Object will be preserved)")    
    result = fill_background_with_sd(
        img, final_mask, fill_prompt, pipe, step=step,
        device=device
    )
    
    return result

# ============================================================================
# MOUSE CALLBACKS
# ============================================================================
def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, bbox, points, sketch_mask
    img_display = param['img_display']
    mode = param['mode']

    # --- BOX MODE ---
    if mode == 'box':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                copy = img_display.copy()
                cv2.rectangle(copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Input Image", cv2.cvtColor(copy, cv2.COLOR_RGB2BGR))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            bbox = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]

    # --- POINT MODE ---
    elif mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            points.append([x, y])

    # --- SKETCH MODE ---
    elif mode == 'sketch':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # Draw on the display image (Red line)
                cv2.line(img_display, (ix, iy), (x, y), (255, 0, 0), 15)
                # Draw on the actual mask (White line on Black)
                if sketch_mask is not None:
                    cv2.line(sketch_mask, (ix, iy), (x, y), 255, 15)
                ix, iy = x, y
                cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Finish the line
            cv2.line(img_display, (ix, iy), (x, y), (255, 0, 0), 15)
            if sketch_mask is not None:
                cv2.line(sketch_mask, (ix, iy), (x, y), 255, 15)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))


def process_with_mask(img, mask, sd_settings, device, output_dir, input_name, dilate_kernel, fill_prompt):
    """Process background filling with a pre-computed mask (used for CLIP/text mode).
    
    Note: The mask represents the OBJECT to KEEP. The background (unmasked area) 
    will be replaced with the generated content.
    """
    timestamp = int(time.time())
    
    # Dilate mask to better protect object edges
    final_mask = dilate_mask(mask, dilate_kernel)
    
    # Show mask overlay (object highlighted in red = area to KEEP)
    mask_overlay = create_mask_overlay(img, final_mask)
    cv2.imshow("Object Mask (area to KEEP)", cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    
    # Fill BACKGROUND with Stable Diffusion (object preserved)
    print(f" -> Filling BACKGROUND with: '{fill_prompt}'...")
    print(f"    (Object will be preserved)")
    fill_start = time.time()
    
    result = fill_background_with_sd(
        img, final_mask, fill_prompt,
        device=device
    )
    fill_time = time.time() - fill_start
    
    # Show filled result
    cv2.imshow("Background Filled Result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # Save minimal results for CLIP mode (mask, masked image, filled only)
    paths = save_minimal_results(img, final_mask, result, output_dir, input_name, timestamp)
    
    # Print timing summary
    print(f"\n{'='*60}")
    print(f"  TIMING SUMMARY - BACKGROUND FILLING")
    print(f"{'='*60}")
    print(f"  Models Used:")
    print(f"    - SAM: MobileSAM (vit_t)")
    print(f"    - SD:  Stable Diffusion + LCM (8 steps)")
    print(f"{'='*60}")
    print(f"  Inference Times:")
    print(f"    - SD Inpainting:     {fill_time:.3f}s")
    print(f"{'='*60}")
    print(f"  Total Inference Time:  {fill_time:.3f}s")
    print(f"{'='*60}")
    print(f"  Background Prompt: '{fill_prompt}'")
    print_saved_results(paths)
    print(f"{'='*60}\n")


def process(img, box=None, points=None, mask_input=None, sam_type="vit_t", sam_ckpt=None,
            sd_settings=None, device="cpu", output_dir="./results", input_name="result", 
            dilate_kernel=15, fill_prompt=""):
    """
    Unified processing function for Box, Point, and Sketch modes.
    Fills the BACKGROUND while keeping the selected object intact.
    
    Args:
        img: Input image (RGB numpy array)
        box: [x1, y1, x2, y2] bounding box for OBJECT to keep
        points: [[x, y], ...] point coordinates on OBJECT to keep
        mask_input: sketch mask for OBJECT to keep
        sam_type: SAM model type
        sam_ckpt: SAM checkpoint path
        sd_settings: Stable Diffusion settings dict
        device: Device to run on
        output_dir: Output directory
        input_name: Input filename stem for output naming
        dilate_kernel: Mask dilation kernel size (protects object edges)
        fill_prompt: Text description for new background
    """
    timestamp = int(time.time())
    total_start = time.time()
    sam_time = 0
    fill_time = 0
    final_mask = None

    # 1. Generate Mask
    if mask_input is not None:
        # Sketch mode: extract bounding box from sketch to use as SAM prompt
        print(" -> Extracting prompt from Sketch...")
        
        # Find contours in sketch mask
        contours, _ = cv2.findContours(mask_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get bounding box of all contours combined
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            sketch_box = [x, y, x + w, y + h]
            
            # Also get center point as additional prompt
            cx, cy = x + w // 2, y + h // 2
            
            print(f" -> Sketch bbox: {sketch_box}, center: ({cx}, {cy})")
            print(" -> Running SAM with sketch prompt...")
            
            # Use both box and center point for better results
            sam_start = time.time()
            masks, scores, _ = predict_masks_with_sam(
                img,
                point_coords=[[cx, cy]],
                point_labels=[1],
                box_coords=np.array(sketch_box),
                model_type=sam_type, ckpt_p=sam_ckpt, device=device
            )
            sam_time = time.time() - sam_start
            
            if len(masks) > 0:
                best_idx = int(np.argmax(scores))
                final_mask = (masks[best_idx] > 0).astype(np.uint8) * 255
            else:
                print(" -> SAM failed, using sketch mask directly.")
                final_mask = mask_input
        else:
            print(" -> No contours found in sketch, using sketch mask directly.")
            final_mask = mask_input
    else:
        # Box/Point mode: use SAM directly
        print(" -> Running SAM...")
        point_coords = points if points else None
        point_labels = [1] * len(points) if points else None
        box_coords = np.array(box) if box else None

        sam_start = time.time()
        masks, scores, _ = predict_masks_with_sam(
            img, point_coords=point_coords, point_labels=point_labels, box_coords=box_coords,
            model_type=sam_type, ckpt_p=sam_ckpt, device=device
        )
        sam_time = time.time() - sam_start
        if len(masks) > 0:
            best_idx = int(np.argmax(scores))
            final_mask = (masks[best_idx] > 0).astype(np.uint8) * 255
        else:
            print("SAM failed.")
            return

    # 2. Dilate Mask (protects object edges from background bleeding)
    final_mask = dilate_mask(final_mask, dilate_kernel)
    
    # 3. Show mask overlay (object = area to KEEP, highlighted in red)
    mask_overlay = create_mask_overlay(img, final_mask)
    cv2.imshow("Object Mask (area to KEEP)", cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    # 4. Fill BACKGROUND with Stable Diffusion (object preserved)
    print(f" -> Filling BACKGROUND with: '{fill_prompt}'...")
    print(f"    (Object will be preserved)")
    fill_start = time.time()
    
    result = fill_background_with_sd(
        img, final_mask, fill_prompt,
        device=device
    )
    fill_time = time.time() - fill_start
    
    total_time = time.time() - total_start
    
    # 5. Show comparison grid
    comparison = create_comparison_grid(img, final_mask, result, mask_overlay)
    cv2.imshow("Comparison", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    cv2.imshow("Background Filled Result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # 6. Save all results using helper function
    paths = save_all_results(img, final_mask, result, output_dir, input_name, timestamp)
    
    # 7. Print timing summary
    print(f"\n{'='*60}")
    print(f"  TIMING SUMMARY - BACKGROUND FILLING")
    print(f"{'='*60}")
    print(f"  Models Used:")
    print(f"    - SAM: MobileSAM (vit_t)")
    print(f"    - SD:  Stable Diffusion + LCM (8 steps)")
    print(f"{'='*60}")
    print(f"  Inference Times:")
    print(f"    - MobileSAM Segmentation:  {sam_time:.3f}s")
    print(f"    - SD Inpainting:           {fill_time:.3f}s")
    print(f"{'='*60}")
    print(f"  Total Inference Time:        {total_time:.3f}s")
    print(f"{'='*60}")
    print(f"  Background Prompt: '{fill_prompt}'")
    print_saved_results(paths)
    print(f"{'='*60}\n")


# ============================================================================
# MAIN LOGIC
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fill background in images using MobileSAM + Stable Diffusion (keeps selected object)")
    parser.add_argument("--input_img", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="./feature1/results/background_filling", help="Output directory for results")
    parser.add_argument("--dilate", type=int, default=7, help="Mask dilation kernel size (protects object edges)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clip_prompt", type=str, default=None, 
                        help="CLIP prompt to find object to KEEP (for text mode). If not provided, will ask interactively.")
    parser.add_argument("--sd_prompt", type=str, default=None,
                        help="Stable Diffusion prompt for new background. If not provided, will ask interactively.")
    args = parser.parse_args()

    # --- Declare globals at the START of main ---
    global sketch_mask, points, bbox

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get input image name for saving results
    input_name = Path(args.input_img).stem

    # Fixed configuration: MobileSAM
    sam_ckpt = "./feature1/checkpoints/mobile_sam.pt"
    sam_model = "vit_t"

    # Setup
    device = "cpu"
    print(f"\n{'='*60}")
    print(f"  Background Filling (MobileSAM + SD with LCM)")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Segmentation Model: MobileSAM (vit_t)")
    print(f"  Inpainting Model:   Stable Diffusion + LCM (8 steps)")
    print(f"  Mode: Replace BACKGROUND, keep selected OBJECT")
    print(f"{'='*60}\n")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load Image
    input_img_path = args.input_img
    original_img = load_img_to_array(input_img_path)
    
    # Initialize Globals
    h, w = original_img.shape[:2]
    sketch_mask = np.zeros((h, w), dtype=np.uint8)  # Blank black mask
    points = []
    bbox = []
    
    # Store SD settings (model loaded on demand)
    sd_settings = {
        'seed': args.seed,
    }
    
    # Display Setup
    img_display = original_img.copy()
    
    cv2.namedWindow("Input Image")
    mouse_params = {'img_display': img_display, 'img_raw': original_img, 'mode': 'none'}
    cv2.setMouseCallback("Input Image", mouse_callback, mouse_params)

    # Store user prompts (from args or will be asked interactively)
    user_clip_prompt = args.clip_prompt
    user_sd_prompt = args.sd_prompt

    print("\nControls:")
    print("  [t] Text | [b] Box | [p] Point | [s] Sketch")
    print("  [Enter] Confirm | [r] Reset | [q] Quit")
    if user_clip_prompt:
        print(f"\nCLIP Prompt (pre-set): '{user_clip_prompt}'")
    if user_sd_prompt:
        print(f"SD Fill Prompt (pre-set): '{user_sd_prompt}'")

    while True:
        cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # --- RESET ---
        elif key == ord('r'):
            img_display[:] = original_img[:]  # Restore clean image in-place
            sketch_mask[:] = 0  # Clear sketch mask in-place
            points.clear()  # Clear points list
            bbox.clear()  # Clear bbox list
            mouse_params['mode'] = 'none'
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            print("Reset. Select a mode: [t] Text | [b] Box | [p] Point | [s] Sketch")

        # --- MODES ---
        elif key == ord('t'):
            if not CLIP_AVAILABLE:
                print("\n>> TEXT MODE not available. Install: pip install open-clip-torch")
            else:
                print("\n>> TEXT MODE (CLIP-based)")
                print("   Select the OBJECT to KEEP (background will be replaced)")
                cv2.waitKey(1)  # Ensure window doesn't freeze
                
                # Ask for CLIP prompt (or use pre-set)
                if user_clip_prompt:
                    text_prompt = user_clip_prompt
                    print(f"   Using pre-set CLIP prompt: '{text_prompt}'")
                else:
                    text_prompt = input("   Enter object description to KEEP (e.g., 'person', 'car'): ").strip()
                
                if not text_prompt:
                    print(" -> No CLIP prompt provided. Cancelling.")
                    print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")
                    continue
                
                print(f" -> Searching for object to keep: '{text_prompt}'...")
                mask = find_object_with_text(original_img, text_prompt, sam_ckpt, device=device)
                
                if mask is not None:
                    # Ask for SD prompt
                    if user_sd_prompt:
                        fill_prompt = user_sd_prompt
                        print(f"   Using pre-set SD prompt: '{fill_prompt}'")
                    else:
                        fill_prompt = input("   Enter NEW BACKGROUND description (e.g., 'a sunny beach', 'city skyline'): ").strip()
                    
                    if fill_prompt:
                        process_with_mask(original_img, mask, sd_settings, device, 
                                        args.output_dir, input_name, args.dilate, fill_prompt)
                else:
                    print(" -> Could not find object. Try different description.")
                print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")

        elif key == ord('s'):
            print("\n>> SKETCH MODE: Draw on the OBJECT to KEEP. Press ENTER when done.")
            mouse_params['mode'] = 'sketch'

        elif key == ord('b'):
            print("\n>> BOX MODE: Draw a box around the OBJECT to KEEP.")
            mouse_params['mode'] = 'box'

        elif key == ord('p'):
            print("\n>> POINT MODE: Click on the OBJECT to KEEP. Press ENTER when done.")
            mouse_params['mode'] = 'point'

        # --- CONFIRM ACTION (Enter/Space) ---
        elif key == 13 or key == 32:
            # Handle Sketch
            if mouse_params['mode'] == 'sketch' and np.max(sketch_mask) > 0:
                cv2.waitKey(1)
                
                # Ask for SD prompt
                if user_sd_prompt:
                    fill_prompt = user_sd_prompt
                    print(f"   Using pre-set SD prompt: '{fill_prompt}'")
                else:
                    fill_prompt = input("   Enter NEW BACKGROUND description (e.g., 'a sunny beach'): ").strip()
                
                if fill_prompt:
                    print("Processing Sketch (keeping object, replacing background)...")
                    process(original_img, mask_input=sketch_mask.copy(), sam_type=sam_model, sam_ckpt=sam_ckpt,
                            sd_settings=sd_settings, device=device, output_dir=args.output_dir,
                            input_name=input_name, dilate_kernel=args.dilate, fill_prompt=fill_prompt)
                
                # Reset after processing
                img_display[:] = original_img[:]
                sketch_mask[:] = 0
                mouse_params['mode'] = 'none'
                print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")

            # Handle Points
            elif mouse_params['mode'] == 'point' and len(points) > 0:
                cv2.waitKey(1)
                
                # Ask for SD prompt
                if user_sd_prompt:
                    fill_prompt = user_sd_prompt
                    print(f"   Using pre-set SD prompt: '{fill_prompt}'")
                else:
                    fill_prompt = input("   Enter NEW BACKGROUND description (e.g., 'a sunny beach'): ").strip()
                
                if fill_prompt:
                    print("Processing Points (keeping object, replacing background)...")
                    process(original_img, points=points.copy(), sam_type=sam_model, sam_ckpt=sam_ckpt,
                            sd_settings=sd_settings, device=device, output_dir=args.output_dir,
                            input_name=input_name, dilate_kernel=args.dilate, fill_prompt=fill_prompt)
                
                # Reset after processing
                img_display[:] = original_img[:]
                points.clear()
                mouse_params['mode'] = 'none'
                print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")

        # --- AUTO-TRIGGER BOX ---
        if mouse_params['mode'] == 'box' and len(bbox) == 4:
            cv2.waitKey(1)
            
            # Ask for SD prompt
            if user_sd_prompt:
                fill_prompt = user_sd_prompt
                print(f"   Using pre-set SD prompt: '{fill_prompt}'")
            else:
                fill_prompt = input("   Enter NEW BACKGROUND description (e.g., 'a sunny beach'): ").strip()
            
            if fill_prompt:
                print("Processing Box (keeping object, replacing background)...")
                process(original_img, box=bbox.copy(), sam_type=sam_model, sam_ckpt=sam_ckpt,
                        sd_settings=sd_settings, device=device, output_dir=args.output_dir,
                        input_name=input_name, dilate_kernel=args.dilate, fill_prompt=fill_prompt)
            
            # Reset after processing
            img_display[:] = original_img[:]
            bbox.clear()
            mouse_params['mode'] = 'none'
            print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()