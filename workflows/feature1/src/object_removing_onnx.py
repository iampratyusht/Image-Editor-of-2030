"""
Object Removal Tool - Compare PyTorch SAM vs ONNX SAM with LAMA FP16

This script allows comparison between:
  - PyTorch MobileSAM + LAMA FP16
  - ONNX MobileSAM + LAMA FP16

Controls:
  Selection Modes:
    [b] - Box Selection
    [p] - Point Selection (Left-click: foreground, Right-click: background)
    [s] - Sketch Selection (Left-click: foreground sketch, Right-click: background sketch)
    [e] - Eraser Mode (erase sketch strokes)
  
  Brush/Eraser Size:
    [+/=] - Increase brush/eraser size
    [-]   - Decrease brush/eraser size
  
  Reset Options:
    [u]   - Undo last point
    [c]   - Clear all points only
    [f]   - Clear foreground sketch only
    [g]   - Clear background sketch only  
    [r]   - Reset everything (all points + all sketches)
  
  Actions:
    [Enter/Space] - Confirm and process selection
    [q] - Quit

Usage:
    # Test with PyTorch SAM
    python object_removing_onnx.py --input_img ./images/3.jpeg --sam_mode pytorch
    
    # Test with ONNX SAM
    python object_removing_onnx.py --input_img ./images/3.jpeg --sam_mode onnx
    
    # Compare both (runs sequentially)
    python object_removing_onnx.py --input_img ./images/3.jpeg --sam_mode compare
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

# LAMA FP16 inference
from .lama_fp16 import LamaFP16

# PyTorch SAM
from .mobilesamsegment import sam_model_registry, SamPredictor

from ..utils import (
    load_img_to_array, save_array_to_img, dilate_mask,
    create_mask_overlay, create_comparison_grid, 
    save_all_results, print_saved_results
)


# --- PATHS ---
SAM_CHECKPOINT = "./feature1/checkpoints/mobile_sam.pt"
SAM_MODEL_TYPE = "vit_t"
ONNX_ENCODER = "./feature1/onnx_export/sam_image_encoder.onnx"
ONNX_POINT_DECODER = "./feature1/onnx_export/sam_mask_decoder_point.onnx"
ONNX_BOX_DECODER = "./feature1/onnx_export/sam_mask_decoder_box.onnx"
LAMA_CHECKPOINT = "./feature1/checkpoints/lama-dilated"
LAMA_FP16_CHECKPOINT = "./feature1/checkpoints/lama-dilated-fp16"


# --- GLOBAL VARIABLES ---
drawing = False
ix, iy = -1, -1
bbox = []
points = []
point_labels = []  # 1=foreground, 0=background
fg_sketch_mask = None  # Foreground sketch
bg_sketch_mask = None  # Background sketch
brush_size = 10  # Default brush size for sketching
eraser_size = 15  # Default eraser size


class ResizeLongestSide:
    """Resize image to have longest side equal to target_length."""
    
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image with longest side = target_length."""
        h, w = image.shape[:2]
        scale = self.target_length / max(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray, original_size: tuple) -> np.ndarray:
        """Transform coordinates from original to resized image space."""
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)
        coords = coords.astype(float).copy()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: tuple) -> np.ndarray:
        """Transform boxes from original to resized image space."""
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple:
        """Compute output size given input size and target long side length."""
        scale = long_side_length / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
        return (newh, neww)


class SAMOnnxInference:
    """ONNX-based SAM inference for mobile deployment."""
    
    def __init__(self, encoder_path: str, point_decoder_path: str, box_decoder_path: str = None, device: str = "cpu"):
        """
        Initialize ONNX SAM inference.
        
        Args:
            encoder_path: Path to image encoder ONNX model
            point_decoder_path: Path to point-based mask decoder ONNX model  
            box_decoder_path: Path to box-based mask decoder ONNX model (optional)
            device: "cpu" or "cuda"
        """
        self.transform = ResizeLongestSide(1024)
        self.img_size = 1024
        
        # Set up ONNX Runtime (CPU optimized for mobile deployment)
        providers = ['CPUExecutionProvider']
        
        print(f"  Loading ONNX encoder: {encoder_path}")
        self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        
        print(f"  Loading ONNX point decoder: {point_decoder_path}")
        self.point_decoder_session = ort.InferenceSession(point_decoder_path, providers=providers)
        
        # Load box decoder if provided
        self.box_decoder_session = None
        if box_decoder_path and os.path.exists(box_decoder_path):
            print(f"  Loading ONNX box decoder: {box_decoder_path}")
            self.box_decoder_session = ort.InferenceSession(box_decoder_path, providers=providers)
        
        # Cache for image embeddings
        self._image_embeddings = None
        self._original_size = None
        self._input_size = None
        
    def set_image(self, image: np.ndarray):
        """
        Preprocess image and compute embeddings.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        self._original_size = image.shape[:2]
        
        # Resize image
        input_image = self.transform.apply_image(image)
        self._input_size = input_image.shape[:2]
        
        # Pad to 1024x1024
        h, w = input_image.shape[:2]
        padh = self.img_size - h
        padw = self.img_size - w
        input_image = np.pad(input_image, ((0, padh), (0, padw), (0, 0)), mode='constant')
        
        # Convert to tensor format [B, C, H, W]
        input_tensor = input_image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
        
        # Run encoder
        encoder_outputs = self.encoder_session.run(None, {"image": input_tensor})
        self._image_embeddings = encoder_outputs[0]
        
    def predict(
        self,
        point_coords: np.ndarray = None,
        point_labels: np.ndarray = None,
        box: np.ndarray = None,
    ) -> tuple:
        """
        Predict mask using point or box prompts.
        
        Args:
            point_coords: (N, 2) array of point coordinates (x, y) in original image space
            point_labels: (N,) array of point labels (1=foreground, 0=background)
            box: (4,) array of box coordinates (x1, y1, x2, y2) in original image space
            
        Returns:
            mask: (H, W) binary mask in original image size
            iou: IoU prediction score
        """
        if self._image_embeddings is None:
            raise RuntimeError("Call set_image() first")
        
        # Transform coordinates to resized image space
        if point_coords is not None:
            point_coords_transformed = self.transform.apply_coords(
                point_coords, self._original_size
            )
            # Reshape for model input [1, N, 2]
            point_coords_input = point_coords_transformed.reshape(1, -1, 2).astype(np.float32)
            point_labels_input = point_labels.reshape(1, -1).astype(np.int64)
            
            # Run point decoder
            decoder_outputs = self.point_decoder_session.run(
                None,
                {
                    "image_embeddings": self._image_embeddings,
                    "point_coords": point_coords_input,
                    "point_labels": point_labels_input,
                }
            )
        elif box is not None:
            # Transform box to resized image space
            box_transformed = self.transform.apply_boxes(
                box.reshape(1, 4), self._original_size
            )
            
            if self.box_decoder_session is not None:
                # Use dedicated box decoder
                box_coords_input = box_transformed.reshape(1, 4).astype(np.float32)
                decoder_outputs = self.box_decoder_session.run(
                    None,
                    {
                        "image_embeddings": self._image_embeddings,
                        "box_coords": box_coords_input,
                    }
                )
            else:
                # Fallback: use point decoder with box corners as points
                # This is less accurate but works if box decoder not available
                point_coords_input = box_transformed.reshape(1, 2, 2).astype(np.float32)
                point_labels_input = np.array([[2, 3]], dtype=np.int64)  # Box corner labels
                
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
        
        masks = decoder_outputs[0]  # [1, 1, 256, 256]
        iou = decoder_outputs[1][0, 0]  # Scalar
        
        # Upscale mask to original size
        mask = masks[0, 0]  # [256, 256]
        mask = cv2.resize(mask, (self.img_size, self.img_size))  # [1024, 1024]
        mask = mask[:self._input_size[0], :self._input_size[1]]  # Remove padding
        mask = cv2.resize(mask, (self._original_size[1], self._original_size[0]))  # Original size
        mask = (mask > 0).astype(np.uint8) * 255
        
        return mask, iou
    
    def get_encoder_time(self) -> float:
        """Get last encoder inference time."""
        return getattr(self, '_encoder_time', 0)
    
    def get_decoder_time(self) -> float:
        """Get last decoder inference time."""
        return getattr(self, '_decoder_time', 0)


class SAMPytorchInference:
    """PyTorch-based SAM inference using MobileSAM."""
    
    def __init__(self, checkpoint: str, model_type: str = "vit_t", device: str = "cpu"):
        """
        Initialize PyTorch SAM inference.
        
        Args:
            checkpoint: Path to SAM checkpoint
            model_type: SAM model type ("vit_t" for MobileSAM)
            device: "cpu" or "cuda"
        """
        self.device = device
        
        print(f"  Loading PyTorch SAM ({model_type}): {checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        sam.eval()
        
        self.predictor = SamPredictor(sam)
        self._original_size = None
    
    def set_image(self, image: np.ndarray):
        """
        Set image for prediction.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        self._original_size = image.shape[:2]
        self.predictor.set_image(image)
    
    def predict(
        self,
        point_coords: np.ndarray = None,
        point_labels: np.ndarray = None,
        box: np.ndarray = None,
    ) -> tuple:
        """
        Predict mask using point or box prompts.
        
        Args:
            point_coords: (N, 2) array of point coordinates (x, y)
            point_labels: (N,) array of point labels (1=foreground, 0=background)
            box: (4,) array of box coordinates (x1, y1, x2, y2)
            
        Returns:
            mask: (H, W) binary mask
            iou: IoU prediction score
        """
        with torch.no_grad():
            if point_coords is not None:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )
            elif box is not None:
                masks, scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False
                )
            else:
                raise ValueError("Must provide either point_coords or box")
        
        # Get best mask
        mask = masks[0]  # [H, W]
        iou = scores[0]
        
        # Convert to 255 scale
        mask = (mask > 0).astype(np.uint8) * 255
        
        return mask, iou


def sketch_to_sam_prompts(fg_sketch: np.ndarray = None, bg_sketch: np.ndarray = None, 
                          num_points: int = 10, min_distance: int = 20) -> tuple:
    """
    Convert foreground and background sketch masks to SAM-eligible point prompts.
    
    Args:
        fg_sketch: Binary mask for foreground sketch (255 = sketched area)
        bg_sketch: Binary mask for background sketch (255 = sketched area)
        num_points: Target number of points to extract per sketch
        min_distance: Minimum distance between points
        
    Returns:
        point_coords: List of [x, y] coordinates
        point_labels: List of labels (1=foreground, 0=background)
    """
    all_points = []
    all_labels = []
    
    # Process foreground sketch
    if fg_sketch is not None and np.sum(fg_sketch) > 0:
        fg_points = _extract_points_from_sketch(fg_sketch, num_points, min_distance)
        all_points.extend(fg_points)
        all_labels.extend([1] * len(fg_points))  # Foreground label
    
    # Process background sketch
    if bg_sketch is not None and np.sum(bg_sketch) > 0:
        bg_points = _extract_points_from_sketch(bg_sketch, num_points, min_distance)
        all_points.extend(bg_points)
        all_labels.extend([0] * len(bg_points))  # Background label
    
    return all_points, all_labels


def _extract_points_from_sketch(sketch_mask: np.ndarray, num_points: int = 10, 
                                 min_distance: int = 20) -> list:
    """
    Extract representative points from a single sketch mask.
    
    Uses multiple strategies:
    1. Contour-based sampling (points along the sketch boundary)
    2. Centroid and extremes
    3. Grid-based sampling (evenly spaced points within sketch)
    
    Args:
        sketch_mask: Binary mask from user sketch (255 = sketched area)
        num_points: Target number of points to extract
        min_distance: Minimum distance between points
        
    Returns:
        List of [x, y] coordinates
    """
    if sketch_mask is None or np.sum(sketch_mask) == 0:
        return []
    
    # Ensure binary mask
    binary_mask = (sketch_mask > 127).astype(np.uint8)
    
    points = []
    
    # Strategy 1: Sample from contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_points = []
    for contour in contours:
        # Sample points along contour
        contour = contour.squeeze()
        if len(contour.shape) == 1:
            contour = contour.reshape(1, 2)
        if len(contour) > 0:
            # Sample evenly along contour
            step = max(1, len(contour) // (num_points // 2))
            for i in range(0, len(contour), step):
                contour_points.append(contour[i].tolist())
    
    # Strategy 2: Sample from centroid and extremes
    if len(contours) > 0:
        # Get bounding box center
        all_contour_points = []
        for c in contours:
            squeezed = c.squeeze()
            if squeezed.ndim >= 1 and len(squeezed) > 0:
                if squeezed.ndim == 1:
                    all_contour_points.append(squeezed.reshape(1, 2))
                else:
                    all_contour_points.append(squeezed)
        
        if all_contour_points:
            all_pts = np.vstack(all_contour_points)
            if len(all_pts) > 0:
                # Centroid
                centroid = np.mean(all_pts, axis=0).astype(int).tolist()
                points.append(centroid)
                
                # Extremes (leftmost, rightmost, topmost, bottommost)
                if len(all_pts) >= 4:
                    points.append(all_pts[all_pts[:, 0].argmin()].tolist())  # Leftmost
                    points.append(all_pts[all_pts[:, 0].argmax()].tolist())  # Rightmost
                    points.append(all_pts[all_pts[:, 1].argmin()].tolist())  # Topmost
                    points.append(all_pts[all_pts[:, 1].argmax()].tolist())  # Bottommost
    
    # Strategy 3: Grid sampling within sketch area
    ys, xs = np.where(binary_mask > 0)
    if len(xs) > 0:
        # Sample random points from sketch area
        indices = np.random.choice(len(xs), min(num_points, len(xs)), replace=False)
        for idx in indices:
            points.append([int(xs[idx]), int(ys[idx])])
    
    # Add contour points
    points.extend(contour_points[:num_points])
    
    # Remove duplicates and filter by minimum distance
    filtered_points = []
    for p in points:
        is_far_enough = True
        for fp in filtered_points:
            dist = np.sqrt((p[0] - fp[0])**2 + (p[1] - fp[1])**2)
            if dist < min_distance:
                is_far_enough = False
                break
        if is_far_enough:
            filtered_points.append(p)
        if len(filtered_points) >= num_points:
            break
    
    return filtered_points


def get_sketch_bounding_box(fg_sketch: np.ndarray = None, padding: int = 10) -> list:
    """
    Get bounding box from foreground sketch mask.
    
    Args:
        fg_sketch: Binary mask from user foreground sketch
        padding: Padding around the bounding box
        
    Returns:
        [x1, y1, x2, y2] bounding box or None if no sketch
    """
    if fg_sketch is None or np.sum(fg_sketch) == 0:
        return None
    
    ys, xs = np.where(fg_sketch > 127)
    if len(xs) == 0:
        return None
    
    x1 = max(0, int(np.min(xs)) - padding)
    y1 = max(0, int(np.min(ys)) - padding)
    x2 = min(fg_sketch.shape[1], int(np.max(xs)) + padding)
    y2 = min(fg_sketch.shape[0], int(np.max(ys)) + padding)
    
    return [x1, y1, x2, y2]


def redraw_display(original_img: np.ndarray, points: list, point_labels: list,
                   fg_sketch: np.ndarray, bg_sketch: np.ndarray) -> np.ndarray:
    """
    Redraw the display image with all current annotations.
    
    Args:
        original_img: Original image
        points: List of point coordinates
        point_labels: List of point labels (1=fg, 0=bg)
        fg_sketch: Foreground sketch mask
        bg_sketch: Background sketch mask
        
    Returns:
        Display image with annotations
    """
    img_display = original_img.copy()
    
    # Draw foreground sketch (green, semi-transparent)
    if fg_sketch is not None and np.sum(fg_sketch) > 0:
        fg_overlay = img_display.copy()
        fg_overlay[fg_sketch > 127] = [0, 255, 0]  # Green
        cv2.addWeighted(fg_overlay, 0.4, img_display, 0.6, 0, img_display)
    
    # Draw background sketch (red, semi-transparent)
    if bg_sketch is not None and np.sum(bg_sketch) > 0:
        bg_overlay = img_display.copy()
        bg_overlay[bg_sketch > 127] = [255, 0, 0]  # Red
        cv2.addWeighted(bg_overlay, 0.4, img_display, 0.6, 0, img_display)
    
    # Draw points
    for i, (pt, label) in enumerate(zip(points, point_labels)):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green for fg, Red for bg
        cv2.circle(img_display, (int(pt[0]), int(pt[1])), 5, color, -1)
        # Add small number label
        cv2.putText(img_display, str(i+1), (int(pt[0])+7, int(pt[1])-7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img_display


def predict_masks_with_sam_onnx(
    img: np.ndarray,
    sam_onnx: SAMOnnxInference,
    point_coords: list = None,
    point_labels: list = None,
    box_coords: np.ndarray = None,
) -> tuple:
    """
    Predict masks using ONNX SAM.
    
    Args:
        img: RGB image as numpy array
        sam_onnx: SAMOnnxInference instance
        point_coords: List of [x, y] points
        point_labels: List of labels (1=foreground, 0=background)
        box_coords: Box coordinates [x1, y1, x2, y2]
        
    Returns:
        mask: Binary mask
        score: IoU score
    """
    # Set image and compute embeddings
    encoder_start = time.time()
    sam_onnx.set_image(img)
    encoder_time = time.time() - encoder_start
    
    # Predict mask
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


def predict_masks_with_sam_pytorch(
    img: np.ndarray,
    sam_pytorch: SAMPytorchInference,
    point_coords: list = None,
    point_labels: list = None,
    box_coords: np.ndarray = None,
) -> tuple:
    """
    Predict masks using PyTorch SAM.
    
    Args:
        img: RGB image as numpy array
        sam_pytorch: SAMPytorchInference instance
        point_coords: List of [x, y] points
        point_labels: List of labels (1=foreground, 0=background)
        box_coords: Box coordinates [x1, y1, x2, y2]
        
    Returns:
        mask: Binary mask
        score: IoU score
    """
    # Set image
    encoder_start = time.time()
    sam_pytorch.set_image(img)
    encoder_time = time.time() - encoder_start
    
    # Predict mask
    decoder_start = time.time()
    
    if point_coords is not None and len(point_coords) > 0:
        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)
        mask, score = sam_pytorch.predict(point_coords=point_coords, point_labels=point_labels)
    elif box_coords is not None:
        box_coords = np.array(box_coords, dtype=np.float32)
        mask, score = sam_pytorch.predict(box=box_coords)
    else:
        raise ValueError("Must provide either point_coords or box_coords")
    
    decoder_time = time.time() - decoder_start
    
    print(f"  PyTorch Encoder: {encoder_time*1000:.1f}ms, Decoder: {decoder_time*1000:.1f}ms")
    
    return mask, score, encoder_time, decoder_time


# ============================================================================
# MOUSE CALLBACKS
# ============================================================================
def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, bbox, points, point_labels, fg_sketch_mask, bg_sketch_mask, brush_size, eraser_size
    img_display = param['img_display']
    original_img = param['original_img']
    mode = param['mode']
    img_shape = param.get('img_shape', img_display.shape[:2])

    # --- BOX MODE ---
    if mode == 'box':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                copy = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
                cv2.rectangle(copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Input Image", cv2.cvtColor(copy, cv2.COLOR_RGB2BGR))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            bbox[:] = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]
            img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
            cv2.rectangle(img_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))

    # --- POINT MODE (Left=foreground, Right=background) ---
    elif mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            # Foreground point (green)
            points.append([x, y])
            point_labels.append(1)  # Foreground
            img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            print(f"    + Foreground point #{len(points)} at ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Background point (red)
            points.append([x, y])
            point_labels.append(0)  # Background
            img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            print(f"    - Background point #{len(points)} at ({x}, {y})")

    # --- SKETCH MODE (Left=foreground sketch, Right=background sketch) ---
    elif mode == 'sketch':
        if fg_sketch_mask is None:
            fg_sketch_mask = np.zeros(img_shape, dtype=np.uint8)
        if bg_sketch_mask is None:
            bg_sketch_mask = np.zeros(img_shape, dtype=np.uint8)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            # Foreground sketch (green)
            cv2.circle(fg_sketch_mask, (x, y), brush_size, 255, -1)
            img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            # Background sketch (red)
            cv2.circle(bg_sketch_mask, (x, y), brush_size, 255, -1)
            img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    # Foreground sketch
                    cv2.circle(fg_sketch_mask, (x, y), brush_size, 255, -1)
                elif flags & cv2.EVENT_FLAG_RBUTTON:
                    # Background sketch
                    cv2.circle(bg_sketch_mask, (x, y), brush_size, 255, -1)
                img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
                cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
                
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            drawing = False

    # --- ERASER MODE ---
    elif mode == 'eraser':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            # Erase from both sketch masks
            if fg_sketch_mask is not None:
                cv2.circle(fg_sketch_mask, (x, y), eraser_size, 0, -1)
            if bg_sketch_mask is not None:
                cv2.circle(bg_sketch_mask, (x, y), eraser_size, 0, -1)
            img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if fg_sketch_mask is not None:
                    cv2.circle(fg_sketch_mask, (x, y), eraser_size, 0, -1)
                if bg_sketch_mask is not None:
                    cv2.circle(bg_sketch_mask, (x, y), eraser_size, 0, -1)
                img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
                cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False


def process_compare(
    img: np.ndarray,
    sam_onnx: SAMOnnxInference,
    sam_pytorch: SAMPytorchInference,
    lama_model: LamaFP16,
    device: str,
    output_dir: str,
    input_name: str,
    box: list = None,
    points_list: list = None,
    points_labels: list = None,
    dilate_kernel: int = 7,
):
    """
    Compare ONNX SAM vs PyTorch SAM with LAMA FP16.
    Runs both and prints timing comparison.
    """
    print("\n" + "="*60)
    print("  COMPARING ONNX SAM vs PyTorch SAM")
    print("="*60)
    
    # Run PyTorch SAM
    print("\n--- PyTorch SAM ---")
    pytorch_results = process_inference(
        img, sam_pytorch, lama_model, device, output_dir, input_name,
        box=box, points_list=points_list, points_labels=points_labels,
        dilate_kernel=dilate_kernel, sam_mode="pytorch"
    )
    
    # Run ONNX SAM
    print("\n--- ONNX SAM ---")
    onnx_results = process_inference(
        img, sam_onnx, lama_model, device, output_dir, input_name,
        box=box, points_list=points_list, points_labels=points_labels,
        dilate_kernel=dilate_kernel, sam_mode="onnx"
    )
    
    # Print comparison
    if pytorch_results and onnx_results:
        print("\n" + "="*60)
        print("  COMPARISON SUMMARY")
        print("="*60)
        print(f"  {'Metric':<20} {'PyTorch':<12} {'ONNX':<12} {'Speedup'}")
        print("-"*60)
        
        metrics = [
            ('SAM Encoder', 'sam_encoder_time'),
            ('SAM Decoder', 'sam_decoder_time'),
            ('SAM Total', 'sam_total_time'),
            ('LAMA FP16', 'lama_time'),
            ('Total Pipeline', 'total_time'),
        ]
        
        for name, key in metrics:
            pt = pytorch_results[key] * 1000
            ox = onnx_results[key] * 1000
            speedup = pt / ox if ox > 0 else 0
            print(f"  {name:<20} {pt:>10.1f}ms {ox:>10.1f}ms {speedup:>6.2f}x")
        
        print("-"*60)
        print(f"  {'IoU Score':<20} {pytorch_results['iou_score']:>10.4f} {onnx_results['iou_score']:>10.4f}")
        print("="*60)


def process_inference(
    img: np.ndarray,
    sam_model,  # Can be SAMOnnxInference or SAMPytorchInference
    lama_model: LamaFP16,
    device: str,
    output_dir: str,
    input_name: str,
    box: list = None,
    points_list: list = None,
    points_labels: list = None,
    dilate_kernel: int = 7,
    sam_mode: str = "onnx",
):
    """
    Unified inference function for both ONNX and PyTorch SAM.
    """
    timestamp = int(time.time())
    total_start = time.time()
    
    # 1. Generate Mask with SAM
    print(f" -> Running {sam_mode.upper()} SAM...")
    
    predict_fn = predict_masks_with_sam_onnx if sam_mode == "onnx" else predict_masks_with_sam_pytorch
    
    if box is not None:
        mask, score, enc_time, dec_time = predict_fn(img, sam_model, box_coords=np.array(box))
    elif points_list is not None and len(points_list) > 0:
        if points_labels is None:
            points_labels = [1] * len(points_list)
        mask, score, enc_time, dec_time = predict_fn(
            img, sam_model, point_coords=points_list, point_labels=points_labels
        )
    else:
        print("No prompt provided!")
        return None
    
    sam_time = enc_time + dec_time
    print(f"  IoU Score: {score:.4f}")
    
    # 2. Dilate Mask
    final_mask = dilate_mask(mask, dilate_kernel)
    
    # 3. Show mask overlay
    mask_overlay = create_mask_overlay(img, final_mask)
    cv2.imshow(f"Mask ({sam_mode.upper()})", cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    
    # 4. Inpaint with LAMA FP16
    print(" -> Inpainting with LAMA FP16...")
    inpaint_start = time.time()
    result = lama_model.inpaint(img, final_mask)
    inpaint_time = time.time() - inpaint_start
    
    total_time = time.time() - total_start
    
    # 5. Show results
    comparison = create_comparison_grid(img, final_mask, result, mask_overlay)
    cv2.imshow(f"Result ({sam_mode.upper()})", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # 6. Save results
    suffix = f"_{sam_mode}"
    paths = save_all_results(img, final_mask, result, output_dir, f"{input_name}{suffix}", timestamp)
    
    # 7. Print timing
    print(f"  {sam_mode.upper()}: SAM={sam_time*1000:.1f}ms, LAMA={inpaint_time*1000:.1f}ms, Total={total_time*1000:.1f}ms")
    
    return {
        'sam_encoder_time': enc_time,
        'sam_decoder_time': dec_time,
        'sam_total_time': sam_time,
        'lama_time': inpaint_time,
        'total_time': total_time,
        'iou_score': float(score)
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch SAM vs ONNX SAM with LAMA FP16")
    parser.add_argument("--input_img", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="./feature1/results/object_removing_compare", help="Output directory")
    parser.add_argument("--sam_mode", type=str, default="compare", choices=["pytorch", "onnx", "compare"],
                       help="SAM mode: pytorch, onnx, or compare (both)")
    parser.add_argument("--sam_ckpt", type=str, default=SAM_CHECKPOINT, help="Path to SAM checkpoint")
    parser.add_argument("--encoder_onnx", type=str, default=ONNX_ENCODER, help="Path to SAM encoder ONNX")
    parser.add_argument("--point_decoder_onnx", type=str, default=ONNX_POINT_DECODER, help="Path to SAM point decoder ONNX")
    parser.add_argument("--box_decoder_onnx", type=str, default=ONNX_BOX_DECODER, help="Path to SAM box decoder ONNX")
    parser.add_argument("--lama_fp16", type=str, default=LAMA_FP16_CHECKPOINT, help="Path to LAMA FP16 checkpoint")
    parser.add_argument("--dilate", type=int, default=7, help="Mask dilation kernel size")
    parser.add_argument("--sketch_points", type=int, default=10, help="Number of points to extract from sketch")
    args = parser.parse_args()

    global points, point_labels, bbox, fg_sketch_mask, bg_sketch_mask, brush_size, eraser_size

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get input image name
    input_name = Path(args.input_img).stem

    # Setup device
    device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"  Object Removal - PyTorch SAM vs ONNX SAM + LAMA FP16")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  SAM Mode: {args.sam_mode}")
    print(f"  Dilation: {args.dilate}")
    print(f"{'='*60}\n")
    
    # Load SAM models based on mode
    sam_pytorch = None
    sam_onnx = None
    
    if args.sam_mode in ["pytorch", "compare"]:
        print("Loading PyTorch SAM...")
        pytorch_load_start = time.time()
        sam_pytorch = SAMPytorchInference(args.sam_ckpt, SAM_MODEL_TYPE, device)
        pytorch_load_time = time.time() - pytorch_load_start
        print(f"  PyTorch SAM loaded in {pytorch_load_time:.2f}s")
    
    if args.sam_mode in ["onnx", "compare"]:
        print("Loading ONNX SAM...")
        onnx_load_start = time.time()
        sam_onnx = SAMOnnxInference(args.encoder_onnx, args.point_decoder_onnx, args.box_decoder_onnx, device)
        onnx_load_time = time.time() - onnx_load_start
        print(f"  ONNX SAM loaded in {onnx_load_time:.2f}s")
    
    # Load LAMA FP16
    print("Loading LAMA FP16 model...")
    lama_load_start = time.time()
    
    # Check if FP16 checkpoint exists, otherwise use regular checkpoint
    if os.path.exists(args.lama_fp16):
        lama_model = LamaFP16(args.lama_fp16, device=device)
    else:
        print(f"  FP16 checkpoint not found: {args.lama_fp16}")
        print(f"  Using regular LAMA checkpoint: {LAMA_CHECKPOINT}")
        lama_model = LamaFP16(LAMA_CHECKPOINT, device=device)
    
    lama_load_time = time.time() - lama_load_start
    print(f"  LAMA FP16 loaded in {lama_load_time:.2f}s")
    
    # Load Image
    print(f"\nLoading image: {args.input_img}")
    original_img = load_img_to_array(args.input_img)
    print(f"  Image size: {original_img.shape[:2]}")
    
    # Initialize globals
    points = []
    point_labels = []
    bbox = []
    fg_sketch_mask = None
    bg_sketch_mask = None
    brush_size = 10
    eraser_size = 15
    
    # Display Setup
    img_display = original_img.copy()
    
    cv2.namedWindow("Input Image")
    mouse_params = {
        'img_display': img_display, 
        'original_img': original_img,
        'mode': 'none',
        'img_shape': original_img.shape[:2]
    }
    cv2.setMouseCallback("Input Image", mouse_callback, mouse_params)

    print("\n" + "="*65)
    print("  CONTROLS:")
    print("-"*65)
    print("  Selection Modes:")
    print("    [b] Box        - Draw bounding box around object")
    print("    [p] Point      - Left=foreground, Right=background")
    print("    [s] Sketch     - Left=foreground sketch, Right=background sketch")
    print("    [e] Eraser     - Erase sketch strokes")
    print("-"*65)
    print("  Brush/Eraser Size:")
    print("    [+/=] Increase | [-] Decrease")
    print("-"*65)
    print("  Reset Options:")
    print("    [u] Undo last point")
    print("    [c] Clear all points only")
    print("    [f] Clear foreground sketch only")
    print("    [g] Clear background sketch only")
    print("    [r] Reset ALL (points + sketches + box)")
    print("-"*65)
    print("  Actions:")
    print("    [Enter/Space] Process | [q] Quit")
    print("="*65 + "\n")

    while True:
        cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # --- FULL RESET ---
        elif key == ord('r'):
            img_display[:] = original_img[:]
            points.clear()
            point_labels.clear()
            bbox.clear()
            fg_sketch_mask = None
            bg_sketch_mask = None
            mouse_params['mode'] = 'none'
            cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            print("Reset ALL. Select mode: [b] Box | [p] Point | [s] Sketch")

        # --- UNDO LAST POINT ---
        elif key == ord('u'):
            if len(points) > 0:
                removed_pt = points.pop()
                removed_label = point_labels.pop()
                label_str = "foreground" if removed_label == 1 else "background"
                print(f"  Undo: removed {label_str} point at ({removed_pt[0]}, {removed_pt[1]})")
                img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
                cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            else:
                print("  No points to undo!")

        # --- CLEAR POINTS ONLY ---
        elif key == ord('c'):
            if len(points) > 0:
                points.clear()
                point_labels.clear()
                print("  Cleared all points (sketches preserved)")
                img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
                cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            else:
                print("  No points to clear!")

        # --- CLEAR FOREGROUND SKETCH ---
        elif key == ord('f'):
            if fg_sketch_mask is not None and np.sum(fg_sketch_mask) > 0:
                fg_sketch_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
                print("  Cleared foreground sketch (green)")
                img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
                cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            else:
                print("  No foreground sketch to clear!")

        # --- CLEAR BACKGROUND SKETCH ---
        elif key == ord('g'):
            if bg_sketch_mask is not None and np.sum(bg_sketch_mask) > 0:
                bg_sketch_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
                print("  Cleared background sketch (red)")
                img_display[:] = redraw_display(original_img, points, point_labels, fg_sketch_mask, bg_sketch_mask)
                cv2.imshow("Input Image", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            else:
                print("  No background sketch to clear!")

        # --- MODES ---
        elif key == ord('b'):
            print("\n>> BOX MODE: Draw a box around the object.")
            mouse_params['mode'] = 'box'
            bbox.clear()

        elif key == ord('p'):
            print("\n>> POINT MODE:")
            print("   Left-click  = Foreground (include in mask)")
            print("   Right-click = Background (exclude from mask)")
            print("   [u] to undo last point, [Enter] when done.")
            mouse_params['mode'] = 'point'

        elif key == ord('s'):
            print(f"\n>> SKETCH MODE: Brush size: {brush_size}")
            print("   Left-drag  = Foreground sketch (green)")
            print("   Right-drag = Background sketch (red)")
            print("   [+/-] to change brush size, [Enter] when done.")
            if fg_sketch_mask is None:
                fg_sketch_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
            if bg_sketch_mask is None:
                bg_sketch_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
            mouse_params['mode'] = 'sketch'

        elif key == ord('e'):
            print(f"\n>> ERASER MODE: Eraser size: {eraser_size}")
            print("   Left-drag to erase sketch strokes")
            print("   [+/-] to change eraser size")
            mouse_params['mode'] = 'eraser'

        # --- BRUSH/ERASER SIZE ---
        elif key == ord('+') or key == ord('='):
            if mouse_params['mode'] == 'eraser':
                eraser_size = min(50, eraser_size + 3)
                print(f"  Eraser size: {eraser_size}")
            else:
                brush_size = min(50, brush_size + 2)
                print(f"  Brush size: {brush_size}")
        elif key == ord('-'):
            if mouse_params['mode'] == 'eraser':
                eraser_size = max(5, eraser_size - 3)
                print(f"  Eraser size: {eraser_size}")
            else:
                brush_size = max(3, brush_size - 2)
                print(f"  Brush size: {brush_size}")

        # --- CONFIRM (Enter/Space) ---
        elif key == 13 or key == 32:  # Enter or Space
            # Combine all prompts
            all_points = points.copy()
            all_labels = point_labels.copy()
            
            # Check for sketches
            has_fg_sketch = fg_sketch_mask is not None and np.sum(fg_sketch_mask) > 0
            has_bg_sketch = bg_sketch_mask is not None and np.sum(bg_sketch_mask) > 0
            
            if has_fg_sketch or has_bg_sketch:
                print(f"\nConverting sketches to SAM prompts...")
                sketch_pts, sketch_lbls = sketch_to_sam_prompts(
                    fg_sketch_mask, bg_sketch_mask,
                    num_points=args.sketch_points,
                    min_distance=15
                )
                all_points.extend(sketch_pts)
                all_labels.extend(sketch_lbls)
                fg_count = sum(1 for l in sketch_lbls if l == 1)
                bg_count = sum(1 for l in sketch_lbls if l == 0)
                print(f"  Extracted {len(sketch_pts)} points from sketch ({fg_count} fg, {bg_count} bg)")
            
            # Determine processing params
            box_param = bbox.copy() if len(bbox) == 4 else None
            points_param = all_points if len(all_points) > 0 else None
            labels_param = all_labels if len(all_labels) > 0 else None
            
            # Fallback: use bounding box from foreground sketch
            if box_param is None and points_param is None and has_fg_sketch:
                box_param = get_sketch_bounding_box(fg_sketch_mask)
            
            if box_param is None and points_param is None:
                print("\nNo selection to process! Use [b] Box, [p] Point, or [s] Sketch first.")
                continue
            
            # Visualize all points
            if points_param:
                viz_img = original_img.copy()
                for pt, lbl in zip(points_param, labels_param):
                    color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
                    cv2.circle(viz_img, (int(pt[0]), int(pt[1])), 5, color, -1)
                cv2.imshow("All Points", cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))
            
            # Process based on mode
            if args.sam_mode == "compare":
                # Run both PyTorch and ONNX for comparison
                process_compare(
                    original_img, sam_onnx, sam_pytorch, lama_model, device,
                    args.output_dir, input_name,
                    box=box_param,
                    points_list=points_param,
                    points_labels=labels_param,
                    dilate_kernel=args.dilate
                )
            elif args.sam_mode == "pytorch":
                # PyTorch only
                process_inference(
                    original_img, sam_pytorch, lama_model, device,
                    args.output_dir, input_name,
                    box=box_param,
                    points_list=points_param,
                    points_labels=labels_param,
                    dilate_kernel=args.dilate,
                    sam_mode="pytorch"
                )
            else:  # onnx
                # ONNX only
                process_inference(
                    original_img, sam_onnx, lama_model, device,
                    args.output_dir, input_name,
                    box=box_param,
                    points_list=points_param,
                    points_labels=labels_param,
                    dilate_kernel=args.dilate,
                    sam_mode="onnx"
                )
            
            # Reset after processing
            img_display[:] = original_img[:]
            points.clear()
            point_labels.clear()
            bbox.clear()
            fg_sketch_mask = None
            bg_sketch_mask = None
            mouse_params['mode'] = 'none'
            print("\nReady for next selection. [b] Box | [p] Point | [s] Sketch")

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
