"""
MobileSAM ONNX Export Script

Exports MobileSAM model components to ONNX format for mobile deployment.
The exported models are optimized for CPU inference on mobile devices.

Components exported:
1. Image Encoder (TinyViT) - Encodes image to embeddings
2. Prompt Encoder + Mask Decoder - Takes prompts and produces masks

Usage:
    python export_sam_onnx.py --checkpoint ./checkpoints/mobile_sam.pt --output_dir ./onnx_models
    
    # Test inference
    python export_sam_onnx.py --checkpoint ./checkpoints/mobile_sam.pt --output_dir ./onnx_models --test --image ./images/3.jpeg

Requirements:
    pip install onnx onnxruntime onnxruntime-tools
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model components
from .mobilesamsegment import sam_model_registry, ResizeLongestSide


class SamImageEncoder(nn.Module):
    """Wrapper for SAM image encoder for ONNX export."""
    
    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.img_size = sam_model.image_encoder.img_size
        self.pixel_mean = sam_model.pixel_mean
        self.pixel_std = sam_model.pixel_std
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, 3, H, W] in RGB format, values 0-255
        Returns:
            Image embeddings [B, 256, 64, 64]
        """
        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std
        
        # Pad to square
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        
        # Encode
        return self.image_encoder(x)


class SamOnnxPromptEncoder(nn.Module):
    """ONNX-compatible prompt encoder that avoids in-place indexing."""
    
    def __init__(self, prompt_encoder):
        super().__init__()
        # Copy all necessary components
        self.embed_dim = prompt_encoder.embed_dim
        self.input_image_size = prompt_encoder.input_image_size
        self.image_embedding_size = prompt_encoder.image_embedding_size
        
        # Register buffers for PE layer
        self.register_buffer(
            'pe_matrix',
            prompt_encoder.pe_layer.positional_encoding_gaussian_matrix
        )
        
        # Register point embeddings as buffers
        self.register_buffer('point_embed_0', prompt_encoder.point_embeddings[0].weight)
        self.register_buffer('point_embed_1', prompt_encoder.point_embeddings[1].weight)
        self.register_buffer('point_embed_2', prompt_encoder.point_embeddings[2].weight)
        self.register_buffer('point_embed_3', prompt_encoder.point_embeddings[3].weight)
        self.register_buffer('not_a_point_embed', prompt_encoder.not_a_point_embed.weight)
        self.register_buffer('no_mask_embed', prompt_encoder.no_mask_embed.weight)
        
        # Pre-compute dense PE
        self.register_buffer('dense_pe', prompt_encoder.get_dense_pe())
        
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.pe_matrix
        coords = 2 * 3.141592653589793 * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    
    def _embed_points_onnx(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """ONNX-compatible point embedding without in-place operations."""
        points = points + 0.5  # Shift to center of pixel
        
        if pad:
            # Add padding point and label
            bs = points.shape[0]
            padding_point = torch.zeros((bs, 1, 2), device=points.device, dtype=points.dtype)
            padding_label = torch.full((bs, 1), -1, device=labels.device, dtype=labels.dtype)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        
        # Normalize coordinates
        points_normalized = points.clone()
        points_normalized[..., 0] = points_normalized[..., 0] / self.input_image_size[1]
        points_normalized[..., 1] = points_normalized[..., 1] / self.input_image_size[0]
        
        # Get positional encoding
        point_embedding = self._pe_encoding(points_normalized)  # [B, N, 256]
        
        # Build label embeddings using broadcasting (ONNX-compatible)
        # labels shape: [B, N], we need [B, N, 256]
        bs, num_points = labels.shape
        
        # Create masks for each label type
        is_neg = (labels == 0).unsqueeze(-1).float()  # [B, N, 1]
        is_pos = (labels == 1).unsqueeze(-1).float()  # [B, N, 1]
        is_pad = (labels == -1).unsqueeze(-1).float()  # [B, N, 1]
        
        # Compute label embedding contribution
        label_embedding = (
            is_neg * self.point_embed_0 +  # background
            is_pos * self.point_embed_1 +  # foreground
            is_pad * self.not_a_point_embed  # padding
        )
        
        # For padding points, zero out the PE and only use not_a_point_embed
        pe_mask = (1 - is_pad)  # 1 for real points, 0 for padding
        point_embedding = point_embedding * pe_mask + label_embedding
        
        return point_embedding
    
    def _embed_boxes_onnx(self, boxes: torch.Tensor) -> torch.Tensor:
        """ONNX-compatible box embedding."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)  # [B, 2, 2] - two corners
        
        # Normalize coordinates
        coords_normalized = coords.clone()
        coords_normalized[..., 0] = coords_normalized[..., 0] / self.input_image_size[1]
        coords_normalized[..., 1] = coords_normalized[..., 1] / self.input_image_size[0]
        
        # Get positional encoding
        corner_embedding = self._pe_encoding(coords_normalized)  # [B, 2, 256]
        
        # Add corner embeddings
        corner_embedding = corner_embedding + torch.stack([
            self.point_embed_2.expand(coords.shape[0], -1),
            self.point_embed_3.expand(coords.shape[0], -1)
        ], dim=1)
        
        return corner_embedding
    
    def forward_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode point prompts."""
        bs = points.shape[0]
        
        # Embed points (pad=True since no box)
        sparse_embeddings = self._embed_points_onnx(points, labels, pad=True)
        
        # Dense embeddings (no mask input)
        dense_embeddings = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )
        
        return sparse_embeddings, dense_embeddings
    
    def forward_box(
        self,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode box prompts."""
        bs = boxes.shape[0]
        
        # Embed box
        sparse_embeddings = self._embed_boxes_onnx(boxes)
        
        # Dense embeddings (no mask input)
        dense_embeddings = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )
        
        return sparse_embeddings, dense_embeddings
    
    def get_dense_pe(self) -> torch.Tensor:
        return self.dense_pe


class SamPromptEncoderAndMaskDecoder(nn.Module):
    """Wrapper for SAM prompt encoder + mask decoder for ONNX export."""
    
    def __init__(self, sam_model):
        super().__init__()
        self.onnx_prompt_encoder = SamOnnxPromptEncoder(sam_model.prompt_encoder)
        self.mask_decoder = sam_model.mask_decoder
        self.mask_threshold = sam_model.mask_threshold
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings: [1, 256, 64, 64] from image encoder
            point_coords: [1, N, 2] point coordinates (x, y) in 1024x1024 space
            point_labels: [1, N] point labels (1=foreground, 0=background)
        Returns:
            masks: [1, 1, 256, 256] predicted mask (low resolution)
            iou_predictions: [1, 1] IoU score
        """
        # Use ONNX-compatible prompt encoder
        sparse_embeddings, dense_embeddings = self.onnx_prompt_encoder.forward_points(
            point_coords, point_labels
        )
        
        # Get dense PE
        image_pe = self.onnx_prompt_encoder.get_dense_pe()
        
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return low_res_masks, iou_predictions


class SamPromptEncoderAndMaskDecoderWithBox(nn.Module):
    """Wrapper for SAM with box prompts for ONNX export."""
    
    def __init__(self, sam_model):
        super().__init__()
        self.onnx_prompt_encoder = SamOnnxPromptEncoder(sam_model.prompt_encoder)
        self.mask_decoder = sam_model.mask_decoder
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        box_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings: [1, 256, 64, 64] from image encoder
            box_coords: [1, 4] box coordinates (x1, y1, x2, y2) in 1024x1024 space
        Returns:
            masks: [1, 1, 256, 256] predicted mask
            iou_predictions: [1, 1] IoU score
        """
        # Use ONNX-compatible prompt encoder with box
        sparse_embeddings, dense_embeddings = self.onnx_prompt_encoder.forward_box(box_coords)
        
        image_pe = self.onnx_prompt_encoder.get_dense_pe()
        
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return low_res_masks, iou_predictions


def export_image_encoder(
    sam_model,
    output_path: str,
    opset_version: int = 17,
    use_fp16: bool = False,
) -> str:
    """Export image encoder to ONNX."""
    print(f"\n{'='*60}")
    print("Exporting Image Encoder...")
    print(f"{'='*60}")
    
    encoder = SamImageEncoder(sam_model)
    encoder.eval()
    
    # Dummy input - 1024x1024 RGB image
    dummy_input = torch.randn(1, 3, 1024, 1024)
    
    if use_fp16:
        encoder = encoder.half()
        dummy_input = dummy_input.half()
    
    output_file = os.path.join(output_path, "sam_image_encoder.onnx")
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output path: {output_file}")
    
    torch.onnx.export(
        encoder,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
    )
    
    print(f"  ✓ Image encoder exported successfully")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return output_file


def export_prompt_encoder_mask_decoder(
    sam_model,
    output_path: str,
    opset_version: int = 17,
    use_fp16: bool = False,
) -> str:
    """Export prompt encoder + mask decoder for point prompts."""
    print(f"\n{'='*60}")
    print("Exporting Prompt Encoder + Mask Decoder (Point Prompts)...")
    print(f"{'='*60}")
    
    decoder = SamPromptEncoderAndMaskDecoder(sam_model)
    decoder.eval()
    
    # Dummy inputs
    image_embeddings = torch.randn(1, 256, 64, 64)
    point_coords = torch.randint(0, 1024, (1, 1, 2)).float()
    point_labels = torch.ones(1, 1, dtype=torch.int64)  # Use int64
    
    if use_fp16:
        decoder = decoder.half()
        image_embeddings = image_embeddings.half()
        point_coords = point_coords.half()
    
    output_file = os.path.join(output_path, "sam_mask_decoder_point.onnx")
    
    print(f"  Image embeddings shape: {image_embeddings.shape}")
    print(f"  Point coords shape: {point_coords.shape}")
    print(f"  Point labels shape: {point_labels.shape}")
    print(f"  Output path: {output_file}")
    
    torch.onnx.export(
        decoder,
        (image_embeddings, point_coords, point_labels),
        output_file,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
    )
    
    print(f"  ✓ Mask decoder (point) exported successfully")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return output_file


def export_mask_decoder_box(
    sam_model,
    output_path: str,
    opset_version: int = 17,
    use_fp16: bool = False,
) -> str:
    """Export prompt encoder + mask decoder for box prompts."""
    print(f"\n{'='*60}")
    print("Exporting Prompt Encoder + Mask Decoder (Box Prompts)...")
    print(f"{'='*60}")
    
    decoder = SamPromptEncoderAndMaskDecoderWithBox(sam_model)
    decoder.eval()
    
    # Dummy inputs
    image_embeddings = torch.randn(1, 256, 64, 64)
    box_coords = torch.tensor([[100, 100, 500, 500]]).float()
    
    if use_fp16:
        decoder = decoder.half()
        image_embeddings = image_embeddings.half()
        box_coords = box_coords.half()
    
    output_file = os.path.join(output_path, "sam_mask_decoder_box.onnx")
    
    print(f"  Image embeddings shape: {image_embeddings.shape}")
    print(f"  Box coords shape: {box_coords.shape}")
    print(f"  Output path: {output_file}")
    
    torch.onnx.export(
        decoder,
        (image_embeddings, box_coords),
        output_file,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image_embeddings", "box_coords"],
        output_names=["masks", "iou_predictions"],
    )
    
    print(f"  ✓ Mask decoder (box) exported successfully")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return output_file


def optimize_onnx_for_mobile(onnx_path: str) -> str:
    """Optimize ONNX model for mobile inference."""
    import onnx
    from onnxruntime.transformers import optimizer
    
    print(f"\n  Optimizing {os.path.basename(onnx_path)} for mobile...")
    
    optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")
    
    try:
        # Load and optimize
        model = onnx.load(onnx_path)
        
        # Basic optimizations
        from onnxruntime.transformers.onnx_model import OnnxModel
        onnx_model = OnnxModel(model)
        onnx_model.optimize()
        onnx_model.save_model_to_file(optimized_path)
        
        print(f"  ✓ Optimized model saved: {optimized_path}")
        print(f"  Optimized size: {os.path.getsize(optimized_path) / 1024 / 1024:.2f} MB")
        
        return optimized_path
    except Exception as e:
        print(f"  ⚠ Optimization failed: {e}")
        print(f"  Using original model")
        return onnx_path


def quantize_to_int8(onnx_path: str) -> str:
    """Quantize ONNX model to INT8 for mobile deployment."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"\n  Quantizing {os.path.basename(onnx_path)} to INT8...")
    
    quantized_path = onnx_path.replace(".onnx", "_int8.onnx")
    
    try:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8,
            optimize_model=True,
        )
        
        print(f"  ✓ INT8 quantized model saved: {quantized_path}")
        print(f"  Quantized size: {os.path.getsize(quantized_path) / 1024 / 1024:.2f} MB")
        
        return quantized_path
    except Exception as e:
        print(f"  ⚠ INT8 quantization failed: {e}")
        return None


def verify_onnx_model(onnx_path: str) -> bool:
    """Verify ONNX model is valid."""
    import onnx
    
    print(f"\n  Verifying {os.path.basename(onnx_path)}...")
    
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ✓ Model verification passed")
        return True
    except Exception as e:
        print(f"  ✗ Model verification failed: {e}")
        return False


def test_onnx_inference(
    encoder_path: str,
    decoder_path: str,
    image_path: str,
    output_dir: str,
):
    """Test ONNX inference with a real image."""
    import onnxruntime as ort
    import cv2
    
    print(f"\n{'='*60}")
    print("Testing ONNX Inference...")
    print(f"{'='*60}")
    
    # Load image
    print(f"\n  Loading image: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    print(f"  Original size: {original_size}")
    
    # Resize to 1024x1024
    transform = ResizeLongestSide(1024)
    input_image = transform.apply_image(image)
    input_size = input_image.shape[:2]
    print(f"  Resized size: {input_size}")
    
    # Pad to 1024x1024
    h, w = input_image.shape[:2]
    padh = 1024 - h
    padw = 1024 - w
    input_image = np.pad(input_image, ((0, padh), (0, padw), (0, 0)), mode='constant')
    
    # Convert to tensor format [B, C, H, W]
    input_tensor = input_image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    print(f"  Input tensor shape: {input_tensor.shape}")
    
    # Create ONNX Runtime sessions
    print("\n  Creating ONNX Runtime sessions...")
    
    # Use CPU execution provider for mobile compatibility
    providers = ['CPUExecutionProvider']
    
    encoder_session = ort.InferenceSession(encoder_path, providers=providers)
    decoder_session = ort.InferenceSession(decoder_path, providers=providers)
    
    # Run image encoder
    print("\n  Running image encoder...")
    start_time = time.time()
    
    encoder_outputs = encoder_session.run(
        None,
        {"image": input_tensor}
    )
    image_embeddings = encoder_outputs[0]
    
    encoder_time = time.time() - start_time
    print(f"  ✓ Image embeddings shape: {image_embeddings.shape}")
    print(f"  ✓ Encoder time: {encoder_time*1000:.2f} ms")
    
    # Prepare point prompt - center of original image, transformed to 1024x1024 space
    # Original image center
    orig_center_x = original_size[1] // 2
    orig_center_y = original_size[0] // 2
    print(f"\n  Original image center: ({orig_center_x}, {orig_center_y})")
    
    # Transform to resized image coordinates
    point_coords_orig = np.array([[orig_center_x, orig_center_y]], dtype=np.float32)
    point_coords_transformed = transform.apply_coords(point_coords_orig, original_size)
    
    # Reshape for model input [1, N, 2]
    point_coords = point_coords_transformed.reshape(1, -1, 2).astype(np.float32)
    point_labels = np.array([[1]], dtype=np.int64)  # Use int64 for ONNX compatibility
    
    print(f"  Transformed point coords: {point_coords}")
    print(f"  Point labels: {point_labels}")
    
    print(f"\n  Running mask decoder with point prompt...")
    
    start_time = time.time()
    
    decoder_outputs = decoder_session.run(
        None,
        {
            "image_embeddings": image_embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }
    )
    masks = decoder_outputs[0]
    iou_predictions = decoder_outputs[1]
    
    decoder_time = time.time() - start_time
    print(f"  ✓ Masks shape: {masks.shape}")
    print(f"  ✓ IoU prediction: {iou_predictions[0, 0]:.4f}")
    print(f"  ✓ Decoder time: {decoder_time*1000:.2f} ms")
    
    # Upscale mask to original size
    mask = masks[0, 0]  # [256, 256]
    mask = cv2.resize(mask, (1024, 1024))  # Upscale to 1024x1024
    mask = mask[:input_size[0], :input_size[1]]  # Remove padding
    mask = cv2.resize(mask, (original_size[1], original_size[0]))  # Resize to original
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Save result
    output_path = os.path.join(output_dir, "onnx_inference_result.png")
    
    # Create visualization
    result = image.copy()
    mask_overlay = np.zeros_like(result)
    mask_overlay[mask > 0] = [0, 255, 0]
    result = cv2.addWeighted(result, 0.7, mask_overlay, 0.3, 0)
    
    # Draw point
    cv2.circle(result, (orig_center_x, orig_center_y), 8, (255, 0, 0), -1)
    cv2.circle(result, (orig_center_x, orig_center_y), 10, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"\n  ✓ Result saved: {output_path}")
    
    # Also save the mask
    mask_path = os.path.join(output_dir, "onnx_inference_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"  ✓ Mask saved: {mask_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Inference Summary")
    print(f"{'='*60}")
    print(f"  Total time: {(encoder_time + decoder_time)*1000:.2f} ms")
    print(f"  - Encoder: {encoder_time*1000:.2f} ms")
    print(f"  - Decoder: {decoder_time*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Export MobileSAM to ONNX for mobile deployment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./feature1/checkpoints/mobile_sam.pt",
        help="Path to MobileSAM checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./feature1/onnx_export",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_t",
        choices=["vit_t", "vit_b", "vit_l", "vit_h"],
        help="SAM model type (vit_t for MobileSAM)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export in FP16 precision"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize to INT8 after export"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize ONNX for mobile"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test ONNX inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Test image path for inference testing"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("MobileSAM ONNX Export")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model type: {args.model_type}")
    print(f"  ONNX opset: {args.opset}")
    print(f"  FP16: {args.fp16}")
    print(f"  Quantize INT8: {args.quantize}")
    
    # Load model
    print(f"\n  Loading SAM model...")
    device = "cpu"  # Export on CPU for compatibility
    
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()
    print(f"  ✓ Model loaded successfully")
    
    # Export components
    encoder_path = export_image_encoder(
        sam, args.output_dir, args.opset, args.fp16
    )
    
    decoder_point_path = export_prompt_encoder_mask_decoder(
        sam, args.output_dir, args.opset, args.fp16
    )
    
    decoder_box_path = export_mask_decoder_box(
        sam, args.output_dir, args.opset, args.fp16
    )
    
    # Verify models
    print(f"\n{'='*60}")
    print("Verifying ONNX Models...")
    print(f"{'='*60}")
    
    verify_onnx_model(encoder_path)
    verify_onnx_model(decoder_point_path)
    verify_onnx_model(decoder_box_path)
    
    # Optimize for mobile
    if args.optimize:
        print(f"\n{'='*60}")
        print("Optimizing for Mobile...")
        print(f"{'='*60}")
        
        encoder_path = optimize_onnx_for_mobile(encoder_path)
        decoder_point_path = optimize_onnx_for_mobile(decoder_point_path)
        decoder_box_path = optimize_onnx_for_mobile(decoder_box_path)
    
    # Quantize to INT8
    if args.quantize:
        print(f"\n{'='*60}")
        print("Quantizing to INT8...")
        print(f"{'='*60}")
        
        quantize_to_int8(encoder_path)
        quantize_to_int8(decoder_point_path)
        quantize_to_int8(decoder_box_path)
    
    # Test inference
    if args.test and args.image:
        test_onnx_inference(
            encoder_path,
            decoder_point_path,
            args.image,
            args.output_dir,
        )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"\n  Exported models:")
    
    for f in Path(args.output_dir).glob("*.onnx"):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    - {f.name}: {size_mb:.2f} MB")
    
    print(f"\n  Usage in mobile app:")
    print(f"    1. Load image encoder and run once per image")
    print(f"    2. Load mask decoder (point or box) and run per prompt")
    print(f"    3. Upscale low-res mask to original image size")
    
    print(f"\n  Mobile deployment notes:")
    print(f"    - Use ONNX Runtime Mobile for iOS/Android")
    print(f"    - INT8 models are smallest but may have slight accuracy loss")
    print(f"    - FP16 models balance size and accuracy")
    print(f"    - Image encoder is the slowest part (~200ms on mobile CPU)")


if __name__ == "__main__":
    main()
