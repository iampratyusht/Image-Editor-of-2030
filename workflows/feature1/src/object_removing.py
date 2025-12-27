"""
Interactive Object Removal Tool (Box, Point, Text, & Sketch)

Controls:
  [t] - Text Prompt (CLIP-based object detection)
  [b] - Box Selection
  [p] - Point Selection
  [s] - Sketch/Draw Selection
  [Enter] - Confirm selection (for Sketch/Points)
  [r] - Reset/Clear selection
  [q] - Quit
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
from .inpaint_by_lama import build_lama_model, inpaint_img_with_builded_lama
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
    print("Note: clip_segment.py not found or CLIP not installed. Text mode disabled.")

# --- GLOBAL VARIABLES ---
drawing = False
ix, iy = -1, -1
bbox = []
points = []
sketch_mask = None # Holds the user's drawing

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

# ============================================================================
# MAIN LOGIC
# ============================================================================
# 
def process_with_mask(img, mask, lama_model, device, output_dir, input_name, dilate_kernel=15):
    """Process inpainting with a pre-computed mask (used for CLIP/text mode)."""
    timestamp = int(time.time())
    
    # Dilate mask
    final_mask = dilate_mask(mask, dilate_kernel)
    
    # Show mask overlay
    mask_overlay = create_mask_overlay(img, final_mask)
    cv2.imshow("Mask Overlay", cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    
    # Inpaint with timing
    print(" -> Inpainting...")
    lama_config = "./feature1/lama/configs/prediction/default.yaml"
    inpaint_start = time.time()
    result = inpaint_img_with_builded_lama(lama_model, img, final_mask, lama_config, device=device)
    inpaint_time = time.time() - inpaint_start
    
    # Show inpainted result
    cv2.imshow("Inpainted Result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # Save minimal results for CLIP mode (mask, masked image, inpainted only)
    paths = save_minimal_results(img, final_mask, result, output_dir, input_name, timestamp)
    
    # Print timing summary
    print(f"\n{'='*40}")
    print(f"  TIMING SUMMARY")
    print(f"{'='*40}")
    print(f"  LAMA Inpainting: {inpaint_time:.3f}s")
    print(f"{'='*40}")
    print_saved_results(paths)
    print(f"{'='*40}\n")


def process(img, box=None, points=None, mask_input=None, sam_type="vit_t", sam_ckpt=None, 
            lama_model=None, device="cpu", output_dir="./results", input_name="result", dilate_kernel=15):
    """
    Unified processing function for Box, Point, and Sketch modes.
    Saves all results: mask, masked image, mask overlay, inpainted, comparison grid.
    
    Args:
        img: Input image (RGB numpy array)
        box: [x1, y1, x2, y2] bounding box
        points: [[x, y], ...] point coordinates
        mask_input: sketch mask (will extract bbox/points from it for SAM)
        sam_type: SAM model type
        sam_ckpt: SAM checkpoint path
        lama_model: Pre-loaded LAMA model
        device: Device to run on
        output_dir: Output directory
        input_name: Input filename stem for output naming
        dilate_kernel: Mask dilation kernel size
    """
    timestamp = int(time.time())
    total_start = time.time()
    sam_time = 0
    inpaint_time = 0
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

    # 2. Dilate Mask
    final_mask = dilate_mask(final_mask, dilate_kernel)
    
    # 3. Show mask overlay using helper function
    mask_overlay = create_mask_overlay(img, final_mask)
    cv2.imshow("Mask Overlay", cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    # 4. Inpaint with timing
    print(" -> Inpainting...")
    lama_config = "./feature1/lama/configs/prediction/default.yaml"
    inpaint_start = time.time()
    result = inpaint_img_with_builded_lama(lama_model, img, final_mask, lama_config, device=device)
    inpaint_time = time.time() - inpaint_start
    
    total_time = time.time() - total_start
    
    # 5. Show comparison grid
    comparison = create_comparison_grid(img, final_mask, result, mask_overlay)
    cv2.imshow("Comparison", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    cv2.imshow("Inpainted Result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # 6. Save all results using helper function
    paths = save_all_results(img, final_mask, result, output_dir, input_name, timestamp)
    
    # 7. Print timing summary
    print(f"\n{'='*40}")
    print(f"  TIMING SUMMARY")
    print(f"{'='*40}")
    print(f"  SAM Segmentation: {sam_time:.3f}s")
    print(f"  LAMA Inpainting:  {inpaint_time:.3f}s")
    print(f"  Total Inference:  {total_time:.3f}s")
    print(f"{'='*40}")
    print_saved_results(paths)
    print(f"{'='*40}\n")

def main():
    parser = argparse.ArgumentParser(description="Remove objects from images using MobileSAM + LAMA")
    parser.add_argument("--input_img", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="./feature1/results/object_removing", help="Output directory for results")
    parser.add_argument("--dilate", type=int, default=7, help="Mask dilation kernel size")
    args = parser.parse_args()

    # --- FIX: Declare globals at the START of main ---
    global sketch_mask, points, bbox

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get input image name for saving results
    input_name = Path(args.input_img).stem

    # Fixed configuration: MobileSAM + LAMA-dilated
    sam_ckpt = "./feature1/checkpoints/mobile_sam.pt"
    lama_config = "./feature1/lama/configs/prediction/default.yaml"
    lama_ckpt = "./feature1/checkpoints/lama-dilated"
    sam_model = "vit_t"

    # Setup
    device = "cpu"
    print(f"\n{'='*50}")
    print(f"  Object Removal (MobileSAM + LAMA)")
    print(f"{'='*50}")
    print(f"  Device: {device}")
    print(f"  SAM: MobileSAM (vit_t)")
    print(f"  Inpainter: LAMA-dilated")
    print(f"{'='*50}\n")
    print("Loading models...")
    lama_load_start = time.time()
    lama_model = build_lama_model(lama_config, lama_ckpt, device=device)
    lama_load_time = time.time() - lama_load_start
    print(f"  LAMA loaded in {lama_load_time:.2f}s")
    
    # Load Image
    input_img_path = args.input_img
    original_img = load_img_to_array(input_img_path)
    
    # Initialize Globals
    h, w = original_img.shape[:2]
    sketch_mask = np.zeros((h, w), dtype=np.uint8) # Blank black mask
    points = []
    bbox = []
    
    # Display Setup
    img_display = original_img.copy()
    
    cv2.namedWindow("Input Image")
    mouse_params = {'img_display': img_display, 'img_raw': original_img, 'mode': 'none'}
    cv2.setMouseCallback("Input Image", mouse_callback, mouse_params)

    print("\nControls:")
    print("  [t] Text | [b] Box | [p] Point | [s] Sketch")
    print("  [Enter] Confirm | [r] Reset | [q] Quit")

    while True:
        cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): break

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
                cv2.waitKey(1)  # Ensure window doesn't freeze
                text_prompt = input("   Enter object description (e.g., 'red car on the left'): ").strip()
                if text_prompt:
                    print(f" -> Searching for: '{text_prompt}'...")
                    mask = find_object_with_text(original_img, text_prompt, sam_ckpt, device=device)
                    if mask is not None:
                        process_with_mask(original_img, mask, lama_model, device, args.output_dir, input_name, args.dilate)
                    else:
                        print(" -> Could not find object. Try different description.")
                print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")

        elif key == ord('s'):
            print("\n>> SKETCH MODE: Draw on the image. Press ENTER when done.")
            mouse_params['mode'] = 'sketch'

        elif key == ord('b'):
            print("\n>> BOX MODE: Draw a box.")
            mouse_params['mode'] = 'box'

        elif key == ord('p'):
            print("\n>> POINT MODE: Click points. Press ENTER when done.")
            mouse_params['mode'] = 'point'

        # --- CONFIRM ACTION (Enter/Space) ---
        elif key == 13 or key == 32:
            # Handle Sketch
            if mouse_params['mode'] == 'sketch' and np.max(sketch_mask) > 0:
                print("Processing Sketch...")
                process(original_img, mask_input=sketch_mask.copy(), sam_type=sam_model, sam_ckpt=sam_ckpt, 
                        lama_model=lama_model, device=device, output_dir=args.output_dir, 
                        input_name=input_name, dilate_kernel=args.dilate)
                # Reset after processing
                img_display[:] = original_img[:]
                sketch_mask[:] = 0
                mouse_params['mode'] = 'none'
                print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")

            # Handle Points
            elif mouse_params['mode'] == 'point' and len(points) > 0:
                print("Processing Points...")
                process(original_img, points=points.copy(), sam_type=sam_model, sam_ckpt=sam_ckpt, 
                        lama_model=lama_model, device=device, output_dir=args.output_dir,
                        input_name=input_name, dilate_kernel=args.dilate)
                # Reset after processing
                img_display[:] = original_img[:]
                points.clear()
                mouse_params['mode'] = 'none'
                print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")

        # --- AUTO-TRIGGER BOX ---
        if mouse_params['mode'] == 'box' and len(bbox) == 4:
            print("Processing Box...")
            process(original_img, box=bbox.copy(), sam_type=sam_model, sam_ckpt=sam_ckpt, 
                    lama_model=lama_model, device=device, output_dir=args.output_dir,
                    input_name=input_name, dilate_kernel=args.dilate)
            # Reset after processing
            img_display[:] = original_img[:]
            bbox.clear()
            mouse_params['mode'] = 'none'
            print("\nReady for next selection. [t] Text | [b] Box | [p] Point | [s] Sketch")

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()