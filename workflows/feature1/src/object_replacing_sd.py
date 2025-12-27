"""
Object Replacing using ONNX SAM + Stable Diffusion LCM Inpainting

This script segments objects using ONNX MobileSAM and replaces them
with AI-generated content using Stable Diffusion with LCM-LoRA.

Controls:
  [b] - Box Selection Mode
  [p] - Point Selection Mode  
  [Enter] - Confirm selection (for Points)
  [r] - Reset/Clear selection
  [q] - Quit

Usage:
    python object_replacing_sd.py --input_img ./images/3.jpeg --prompt "a cute puppy"
    
    # With custom seed for reproducibility
    python object_replacing_sd.py --input_img ./images/3.jpeg --prompt "a flower vase" --seed 42
"""

import cv2
import torch
import sys
import time
import numpy as np
import argparse
from pathlib import Path
import os

import onnxruntime as ort

from .inpaint_by_sd import build_sd_inpaint_model, inpaint_with_sd
from ..utils import (
    load_img_to_array, save_array_to_img, dilate_mask,
    create_mask_overlay, create_comparison_grid, 
    save_all_results, print_saved_results
)


# --- GLOBAL VARIABLES ---
drawing = False
ix, iy = -1, -1
bbox = []
points = []

def replace_object(
    img: np.ndarray,
    mask: np.ndarray,
    sd_pipe,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted",
    dilate_kernel: int = 15,
    seed: int = None,
    step: int= 4
):
    # 2. Dilate Mask
    final_mask = dilate_mask(mask, dilate_kernel)

    print(f" -> Replacing with SD (prompt: '{prompt}')...")
    # sd_start = time.time()
    result = inpaint_with_sd(
        sd_pipe, img, final_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=step,
        guidance_scale=4.0,
        seed=seed
    )
    
    return result

class ResizeLongestSide:
    """Resize image to have longest side equal to target_length."""
    
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self.target_length / max(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray, original_size: tuple) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)
        coords = coords.astype(float).copy()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: tuple) -> np.ndarray:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple:
        scale = long_side_length / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
        return (newh, neww)


class SAMOnnxInference:
    """ONNX-based SAM inference."""
    
    def __init__(self, encoder_path: str, point_decoder_path: str, box_decoder_path: str = None, device: str = "cpu"):
        self.transform = ResizeLongestSide(1024)
        self.img_size = 1024
        
        # Set up ONNX Runtime (CPU optimized for mobile deployment)
        providers = ['CPUExecutionProvider']
        
        print(f"  Loading ONNX encoder: {encoder_path}")
        self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        
        print(f"  Loading ONNX point decoder: {point_decoder_path}")
        self.point_decoder_session = ort.InferenceSession(point_decoder_path, providers=providers)
        
        self.box_decoder_session = None
        if box_decoder_path and os.path.exists(box_decoder_path):
            print(f"  Loading ONNX box decoder: {box_decoder_path}")
            self.box_decoder_session = ort.InferenceSession(box_decoder_path, providers=providers)
        
        self._image_embeddings = None
        self._original_size = None
        self._input_size = None
        
    def set_image(self, image: np.ndarray):
        self._original_size = image.shape[:2]
        input_image = self.transform.apply_image(image)
        self._input_size = input_image.shape[:2]
        
        h, w = input_image.shape[:2]
        padh = self.img_size - h
        padw = self.img_size - w
        input_image = np.pad(input_image, ((0, padh), (0, padw), (0, 0)), mode='constant')
        input_tensor = input_image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
        
        encoder_outputs = self.encoder_session.run(None, {"image": input_tensor})
        self._image_embeddings = encoder_outputs[0]
        
    def predict(self, point_coords: np.ndarray = None, point_labels: np.ndarray = None, box: np.ndarray = None) -> tuple:
        if self._image_embeddings is None:
            raise RuntimeError("Call set_image() first")
        
        if point_coords is not None:
            point_coords_transformed = self.transform.apply_coords(point_coords, self._original_size)
            point_coords_input = point_coords_transformed.reshape(1, -1, 2).astype(np.float32)
            point_labels_input = point_labels.reshape(1, -1).astype(np.int64)
            
            decoder_outputs = self.point_decoder_session.run(
                None,
                {
                    "image_embeddings": self._image_embeddings,
                    "point_coords": point_coords_input,
                    "point_labels": point_labels_input,
                }
            )
        elif box is not None:
            box_transformed = self.transform.apply_boxes(box.reshape(1, 4), self._original_size)
            
            if self.box_decoder_session is not None:
                box_coords_input = box_transformed.reshape(1, 4).astype(np.float32)
                decoder_outputs = self.box_decoder_session.run(
                    None,
                    {
                        "image_embeddings": self._image_embeddings,
                        "box_coords": box_coords_input,
                    }
                )
            else:
                point_coords_input = box_transformed.reshape(1, 2, 2).astype(np.float32)
                point_labels_input = np.array([[2, 3]], dtype=np.int64)
                decoder_outputs = self.point_decoder_session.run(
                    None,
                    {
                        "image_embeddings": self._image_embeddings,
                        "point_coords": point_coords_input,
                        "point_labels": point_labels_input,
                    }
                )
        else:
            raise ValueError("Must provide either point_coords or box")
        
        masks = decoder_outputs[0]
        iou = decoder_outputs[1][0, 0]
        
        mask = masks[0, 0]
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = mask[:self._input_size[0], :self._input_size[1]]
        mask = cv2.resize(mask, (self._original_size[1], self._original_size[0]))
        mask = (mask > 0).astype(np.uint8) * 255
        
        return mask, iou


def predict_masks_with_sam_onnx(img, sam_onnx, point_coords=None, point_labels=None, box_coords=None):
    encoder_start = time.time()
    sam_onnx.set_image(img)
    encoder_time = time.time() - encoder_start
    
    decoder_start = time.time()
    
    if point_coords is not None and len(point_coords) > 0:
        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int64)
        mask, score = sam_onnx.predict(point_coords=point_coords, point_labels=point_labels)
    elif box_coords is not None:
        box_coords = np.array(box_coords, dtype=np.float32)
        mask, score = sam_onnx.predict(box=box_coords)
    else:
        raise ValueError("Must provide either point_coords or box_coords")
    
    decoder_time = time.time() - decoder_start
    print(f"  ONNX Encoder: {encoder_time*1000:.1f}ms, Decoder: {decoder_time*1000:.1f}ms")
    
    return mask, score, encoder_time, decoder_time


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, bbox, points
    img_display = param['img_display']
    mode = param['mode']

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

    elif mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            points.append([x, y])


def process_replacement(
    img: np.ndarray,
    sam_onnx: SAMOnnxInference,
    sd_pipe,
    device: str,
    output_dir: str,
    input_name: str,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted",
    box: list = None,
    points_list: list = None,
    dilate_kernel: int = 3,
    seed: int = None,
):
    """Process segmentation and replacement using ONNX SAM + Stable Diffusion."""
    timestamp = int(time.time())
    total_start = time.time()
    
    # 1. Generate Mask with ONNX SAM
    print(" -> Running ONNX SAM...")
    
    if box is not None:
        mask, score, enc_time, dec_time = predict_masks_with_sam_onnx(
            img, sam_onnx, box_coords=np.array(box)
        )
    elif points_list is not None and len(points_list) > 0:
        point_labels = [1] * len(points_list)
        mask, score, enc_time, dec_time = predict_masks_with_sam_onnx(
            img, sam_onnx, point_coords=points_list, point_labels=point_labels
        )
    else:
        print("No prompt provided!")
        return
    
    sam_time = enc_time + dec_time
    print(f"  IoU Score: {score:.4f}")
    
    # 2. Dilate Mask
    final_mask = dilate_mask(mask, dilate_kernel)
    
    # 3. Show mask overlay
    mask_overlay = create_mask_overlay(img, final_mask)
    cv2.imshow("Mask Overlay", cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    
    # 4. Replace with Stable Diffusion
    print(f" -> Replacing with SD (prompt: '{prompt}')...")
    sd_start = time.time()
    result = inpaint_with_sd(
        sd_pipe, img, final_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=8,
        guidance_scale=4.0,
        seed=seed
    )
    sd_time = time.time() - sd_start
    
    total_time = time.time() - total_start
    
    # 5. Show results
    comparison = create_comparison_grid(img, final_mask, result, mask_overlay)
    cv2.imshow("Comparison", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    cv2.imshow("Replaced Result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # 6. Save results
    paths = save_all_results(img, final_mask, result, output_dir, input_name, timestamp)
    
    # 7. Print timing summary
    print(f"\n{'='*50}")
    print(f"  OBJECT REPLACEMENT TIMING SUMMARY")
    print(f"{'='*50}")
    print(f"  ONNX SAM Encoder:  {enc_time*1000:.1f} ms")
    print(f"  ONNX SAM Decoder:  {dec_time*1000:.1f} ms")
    print(f"  SAM Total:         {sam_time*1000:.1f} ms")
    print(f"  SD Inpainting:     {sd_time*1000:.1f} ms")
    print(f"  Total Pipeline:    {total_time*1000:.1f} ms")
    print(f"{'='*50}")
    print_saved_results(paths)
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Replace objects using ONNX SAM + Stable Diffusion")
    parser.add_argument("--input_img", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="a beautiful object, high quality, detailed", help="What to replace the object with")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted", help="What to avoid")
    parser.add_argument("--output_dir", type=str, default="./feature1/results/object_replacing_sd", help="Output directory")
    parser.add_argument("--encoder_onnx", type=str, default="./feature1/onnx_export/sam_image_encoder.onnx")
    parser.add_argument("--point_decoder_onnx", type=str, default="./feature1/onnx_export/sam_mask_decoder_point.onnx")
    parser.add_argument("--box_decoder_onnx", type=str, default="./feature1/onnx_export/sam_mask_decoder_box.onnx")
    parser.add_argument("--dilate", type=int, default=3, help="Mask dilation kernel size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    global points, bbox

    os.makedirs(args.output_dir, exist_ok=True)
    input_name = Path(args.input_img).stem
    device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"  Object Replacement (ONNX SAM + Stable Diffusion LCM)")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    
    # Load ONNX SAM
    print("Loading ONNX SAM models...")
    sam_onnx = SAMOnnxInference(args.encoder_onnx, args.point_decoder_onnx, args.box_decoder_onnx, device)
    
    # Load Stable Diffusion
    print("\nLoading Stable Diffusion with LCM...")
    sd_pipe = build_sd_inpaint_model(device=device, use_lcm=True)
    
    # Load Image
    print(f"\nLoading image: {args.input_img}")
    original_img = load_img_to_array(args.input_img)
    print(f"  Image size: {original_img.shape[:2]}")
    
    points = []
    bbox = []
    img_display = original_img.copy()
    
    cv2.namedWindow("Input Image")
    mouse_params = {'img_display': img_display, 'mode': 'none'}
    cv2.setMouseCallback("Input Image", mouse_callback, mouse_params)

    print("\n" + "="*60)
    print("  Controls:")
    print("  [b] Box Mode | [p] Point Mode")
    print("  [Enter] Confirm Points | [r] Reset | [q] Quit")
    print("="*60 + "\n")

    while True:
        cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            img_display[:] = original_img[:]
            points.clear()
            bbox.clear()
            mouse_params['mode'] = 'none'
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            print("Reset. Select mode: [b] Box | [p] Point")

        elif key == ord('b'):
            print("\n>> BOX MODE: Draw a box around the object to replace.")
            mouse_params['mode'] = 'box'

        elif key == ord('p'):
            print("\n>> POINT MODE: Click points on the object. Press ENTER when done.")
            mouse_params['mode'] = 'point'

        elif key == 13 or key == 32:  # Enter or Space
            if mouse_params['mode'] == 'point' and len(points) > 0:
                print(f"\nProcessing {len(points)} point(s)...")
                process_replacement(
                    original_img, sam_onnx, sd_pipe, device,
                    args.output_dir, input_name,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    points_list=points.copy(),
                    dilate_kernel=args.dilate,
                    seed=args.seed
                )
                img_display[:] = original_img[:]
                points.clear()
                mouse_params['mode'] = 'none'
                print("\nReady for next selection. [b] Box | [p] Point")

        if mouse_params['mode'] == 'box' and len(bbox) == 4:
            print(f"\nProcessing box: {bbox}")
            process_replacement(
                original_img, sam_onnx, sd_pipe, device,
                args.output_dir, input_name,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                box=bbox.copy(),
                dilate_kernel=args.dilate,
                seed=args.seed
            )
            img_display[:] = original_img[:]
            bbox.clear()
            mouse_params['mode'] = 'none'
            print("\nReady for next selection. [b] Box | [p] Point")

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
