"""
LaMa FP16 Model - Quantized for CPU Inference and Flask Deployment

This module provides:
1. FP16 weight quantization for LaMa model
2. CPU-optimized inference with FP16 weights (computation in FP32)
3. Flask-ready API for deployment

The model stores weights in FP16 for 50% memory reduction but computes in FP32
for CPU compatibility (most CPUs don't support FP16 natively).

Usage:
    # Quantize model to FP16
    python lama_fp16.py --quantize --input_dir ./checkpoints/lama-dilated --output_dir ./checkpoints/lama-dilated-fp16
    
    # Run inference
    python lama_fp16.py --infer --image ./images/test.jpg --mask ./masks/test.png --checkpoint_dir ./checkpoints/lama-dilated-fp16
    
    # Start Flask server
    python lama_fp16.py --serve --port 5000 --checkpoint_dir ./checkpoints/lama-dilated-fp16
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Union

# Add lama to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_DIR = os.path.join(SCRIPT_DIR, 'lama')
if os.path.exists(LAMA_DIR):
    sys.path.insert(0, LAMA_DIR)

from omegaconf import OmegaConf


# =============================================================================
# FP16 WEIGHT QUANTIZATION
# =============================================================================

def quantize_state_dict_to_fp16(state_dict: dict) -> dict:
    """
    Quantize all FP32 tensors in state dict to FP16.
    
    Args:
        state_dict: Original state dict with FP32 weights
        
    Returns:
        State dict with FP16 weights
    """
    fp16_state_dict = {}
    fp32_count = 0
    fp16_count = 0
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float32:
                fp16_state_dict[key] = value.half()
                fp32_count += 1
                fp16_count += 1
            else:
                fp16_state_dict[key] = value
        else:
            fp16_state_dict[key] = value
    
    print(f"  Converted {fp32_count} FP32 tensors to FP16")
    return fp16_state_dict


def get_model_size_mb(state_dict: dict) -> float:
    """Calculate total size of model in MB."""
    total_size = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total_size += value.numel() * value.element_size()
    return total_size / (1024 * 1024)


def quantize_lama_checkpoint(
    input_dir: str,
    output_dir: str,
    checkpoint_name: str = "best.ckpt"
) -> str:
    """
    Quantize LaMa checkpoint weights to FP16.
    
    Args:
        input_dir: Directory containing models/best.ckpt and config.yaml
        output_dir: Output directory for FP16 model
        checkpoint_name: Name of checkpoint file
        
    Returns:
        Path to quantized checkpoint
    """
    import shutil
    
    print(f"\n{'='*60}")
    print("  LaMa FP16 Weight Quantization")
    print(f"{'='*60}")
    
    input_ckpt = os.path.join(input_dir, "models", checkpoint_name)
    input_config = os.path.join(input_dir, "config.yaml")
    output_ckpt = os.path.join(output_dir, "models", checkpoint_name)
    output_config = os.path.join(output_dir, "config.yaml")
    
    # Check input exists
    if not os.path.exists(input_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {input_ckpt}")
    
    # Load checkpoint (weights_only=False needed for PyTorch Lightning checkpoints)
    print(f"\nLoading: {input_ckpt}")
    checkpoint = torch.load(input_ckpt, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        is_full_checkpoint = True
    else:
        state_dict = checkpoint
        is_full_checkpoint = False
    
    original_size = get_model_size_mb(state_dict)
    print(f"  Original size: {original_size:.2f} MB")
    
    # Quantize
    print("\nQuantizing weights to FP16...")
    fp16_state_dict = quantize_state_dict_to_fp16(state_dict)
    
    new_size = get_model_size_mb(fp16_state_dict)
    print(f"  New size: {new_size:.2f} MB")
    print(f"  Reduction: {(1 - new_size/original_size)*100:.1f}%")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    
    # Save quantized checkpoint
    if is_full_checkpoint:
        checkpoint['state_dict'] = fp16_state_dict
        torch.save(checkpoint, output_ckpt)
    else:
        torch.save(fp16_state_dict, output_ckpt)
    
    print(f"\nSaved: {output_ckpt}")
    
    # Copy config
    if os.path.exists(input_config):
        os.makedirs(os.path.dirname(output_config), exist_ok=True)
        shutil.copy(input_config, output_config)
        print(f"Copied config: {output_config}")
    
    print(f"\n{'='*60}")
    print("  Quantization Complete!")
    print(f"{'='*60}\n")
    
    return output_ckpt


# =============================================================================
# LAMA FP16 MODEL WRAPPER
# =============================================================================

class LamaFP16:
    """
    LaMa model with FP16 weights for memory-efficient CPU inference.
    
    Weights are stored in FP16, but computation happens in FP32 for CPU compatibility.
    This gives ~50% memory reduction while maintaining inference quality.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        config_path: str = None,
        device: str = "cpu",
        use_fp16_weights: bool = True
    ):
        """
        Initialize LaMa model with FP16 weights.
        
        Args:
            checkpoint_dir: Directory containing models/best.ckpt and config.yaml
            config_path: Path to prediction config (optional)
            device: Device to run on ("cpu" or "cuda")
            use_fp16_weights: Whether to convert weights to FP16 on load
        """
        self.device = device
        self.use_fp16_weights = use_fp16_weights
        self.model = None
        self.predict_config = None
        
        self._load_model(checkpoint_dir, config_path)
    
    def _load_model(self, checkpoint_dir: str, config_path: str = None):
        """Load and prepare the model."""
        from lama.saicinpainting.training.trainers import load_checkpoint
        
        if config_path is None:
            config_path = os.path.join(SCRIPT_DIR, "lama", "configs", "prediction", "default.yaml")
        
        # Load prediction config
        self.predict_config = OmegaConf.load(config_path)
        self.predict_config.model.path = checkpoint_dir
        
        # Load training config
        train_config_path = os.path.join(checkpoint_dir, 'config.yaml')
        if not os.path.exists(train_config_path):
            raise FileNotFoundError(f"Config not found: {train_config_path}")
        
        train_config = OmegaConf.load(train_config_path)
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        
        # Load checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, 'models',
            self.predict_config.model.checkpoint
        )
        
        print(f"Loading LaMa model from: {checkpoint_path}")
        load_start = time.time()
        
        self.model = load_checkpoint(
            train_config,
            checkpoint_path,
            strict=False,
            map_location='cpu'
        )
        self.model.freeze()
        
        # Convert weights to FP16 if requested (for memory savings)
        if self.use_fp16_weights:
            self._convert_weights_to_fp16()
        
        self.model = self.model.to(self.device)
        
        load_time = time.time() - load_start
        print(f"  Model loaded in {load_time:.2f}s")
    
    def _convert_weights_to_fp16(self):
        """Convert model weights to FP16 for memory efficiency."""
        print("  Converting weights to FP16 for memory efficiency...")
        for param in self.model.parameters():
            param.data = param.data.half()
    
    def _pad_to_modulo(self, tensor: torch.Tensor, mod: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad tensor to be divisible by mod."""
        _, _, h, w = tensor.shape
        pad_h = (mod - h % mod) % mod
        pad_w = (mod - w % mod) % mod
        
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        return tensor, (h, w)
    
    @torch.no_grad()
    def inpaint(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Inpaint image using the mask.
        
        Args:
            image: RGB image (H, W, 3) uint8 numpy array or (B, 3, H, W) tensor
            mask: Binary mask (H, W) where 255/1 = area to inpaint
            
        Returns:
            Inpainted image as (H, W, 3) uint8 numpy array
        """
        from lama.saicinpainting.evaluation.utils import move_to_device
        
        # Convert numpy to tensor
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=2)
            image_tensor = torch.from_numpy(image).float().div(255.0)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image
        
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.max() > 1:
                mask = mask / 255.0
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        else:
            mask_tensor = mask
        
        # Pad to modulo 8
        image_padded, original_size = self._pad_to_modulo(image_tensor, 8)
        mask_padded, _ = self._pad_to_modulo(mask_tensor, 8)
        
        # Create batch
        batch = {
            'image': image_padded,
            'mask': mask_padded
        }
        
        # Move to device and convert to FP32 for computation
        batch = move_to_device(batch, self.device)
        batch['image'] = batch['image'].float()  # Ensure FP32 for computation
        batch['mask'] = (batch['mask'] > 0).float()
        
        # Run inference
        result = self.model(batch)
        
        # Get output
        output = result['inpainted'][0].permute(1, 2, 0)
        output = output.cpu().numpy()
        
        # Unpad
        h, w = original_size
        output = output[:h, :w]
        
        # Convert to uint8
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        return output
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Convenience method for inpainting."""
        return self.inpaint(image, mask)


# =============================================================================
# GLOBAL MODEL INSTANCE FOR FLASK
# =============================================================================

_lama_model: Optional[LamaFP16] = None


def get_lama_model(
    checkpoint_dir: str = "./feature1/checkpoints/lama-dilated",
    device: str = "cpu",
    use_fp16: bool = True
) -> LamaFP16:
    """
    Get or create global LaMa model instance (singleton pattern for Flask).
    
    Args:
        checkpoint_dir: Path to LaMa checkpoint directory
        device: Device to run on
        use_fp16: Whether to use FP16 weights
        
    Returns:
        LamaFP16 model instance
    """
    global _lama_model
    
    if _lama_model is None:
        print("Initializing LaMa FP16 model...")
        _lama_model = LamaFP16(
            checkpoint_dir=checkpoint_dir,
            device=device,
            use_fp16_weights=use_fp16
        )
    
    return _lama_model


def inpaint_with_lama_fp16(
    image: np.ndarray,
    mask: np.ndarray,
    checkpoint_dir: str = "./feature1/checkpoints/lama-dilated",
    device: str = "cpu"
) -> np.ndarray:
    """
    Convenience function for inpainting with LaMa FP16.
    
    Args:
        image: RGB image (H, W, 3) uint8
        mask: Binary mask (H, W) where 255 = area to inpaint
        checkpoint_dir: Path to LaMa checkpoint
        device: Device to run on
        
    Returns:
        Inpainted image (H, W, 3) uint8
    """
    model = get_lama_model(checkpoint_dir, device)
    return model.inpaint(image, mask)


# =============================================================================
# FLASK API
# =============================================================================

def create_flask_app(
    checkpoint_dir: str = "./feature1/checkpoints/lama-dilated",
    device: str = "cpu",
    use_fp16: bool = True
):
    """
    Create Flask app for LaMa inpainting API.
    
    Endpoints:
        POST /inpaint - Inpaint image with mask
            Body: multipart/form-data with 'image' and 'mask' files
            Returns: Inpainted image as PNG
        
        GET /health - Health check
            Returns: {"status": "ok", "model": "lama-fp16"}
    """
    from flask import Flask, request, jsonify, send_file
    from io import BytesIO
    from PIL import Image
    
    app = Flask(__name__)
    
    # Initialize model on startup
    print("Initializing LaMa FP16 for Flask...")
    model = get_lama_model(checkpoint_dir, device, use_fp16)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "ok",
            "model": "lama-fp16",
            "device": device,
            "fp16_weights": use_fp16
        })
    
    @app.route('/inpaint', methods=['POST'])
    def inpaint():
        try:
            # Check for required files
            if 'image' not in request.files or 'mask' not in request.files:
                return jsonify({"error": "Missing 'image' or 'mask' file"}), 400
            
            # Read image
            image_file = request.files['image']
            image = Image.open(image_file).convert('RGB')
            image = np.array(image)
            
            # Read mask
            mask_file = request.files['mask']
            mask = Image.open(mask_file).convert('L')
            mask = np.array(mask)
            
            # Inpaint
            start_time = time.time()
            result = model.inpaint(image, mask)
            inference_time = time.time() - start_time
            
            # Convert to PNG bytes
            result_pil = Image.fromarray(result)
            img_io = BytesIO()
            result_pil.save(img_io, 'PNG')
            img_io.seek(0)
            
            # Add inference time to response headers
            response = send_file(img_io, mimetype='image/png')
            response.headers['X-Inference-Time'] = f"{inference_time:.3f}s"
            
            return response
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LaMa FP16 - Quantization, Inference, and Flask Server")
    
    # Mode selection
    parser.add_argument("--quantize", action="store_true", help="Quantize model weights to FP16")
    parser.add_argument("--infer", action="store_true", help="Run inference on an image")
    parser.add_argument("--serve", action="store_true", help="Start Flask server")
    
    # Paths
    parser.add_argument("--input_dir", type=str, default="./feature1/checkpoints/lama-dilated",
                       help="Input LaMa checkpoint directory (for quantize)")
    parser.add_argument("--output_dir", type=str, default="./feature1/checkpoints/lama-dilated-fp16",
                       help="Output directory for FP16 model (for quantize)")
    parser.add_argument("--checkpoint_dir", type=str, default="./feature1/checkpoints/lama-dilated",
                       help="LaMa checkpoint directory (for infer/serve)")
    
    # Inference options
    parser.add_argument("--image", type=str, help="Input image path (for infer)")
    parser.add_argument("--mask", type=str, help="Mask image path (for infer)")
    parser.add_argument("--output", type=str, default=None, help="Output path (for infer)")
    
    # Server options
    parser.add_argument("--port", type=int, default=5000, help="Flask server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Flask server host")
    
    # Common options
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 weights")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Don't use FP16 weights")
    
    args = parser.parse_args()
    
    # === QUANTIZE MODE ===
    if args.quantize:
        quantize_lama_checkpoint(args.input_dir, args.output_dir)
        return
    
    # === INFERENCE MODE ===
    if args.infer:
        if not args.image or not args.mask:
            parser.error("--infer requires --image and --mask")
        
        print(f"\n{'='*60}")
        print(f"  LaMa FP16 Inference")
        print(f"{'='*60}")
        print(f"  Device: {args.device}")
        print(f"  FP16 Weights: {args.fp16}")
        print(f"  Checkpoint: {args.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        # Load image and mask
        print(f"Loading image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image: {args.image}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"Loading mask: {args.mask}")
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not load mask: {args.mask}")
            return
        
        print(f"  Image size: {image.shape[:2]}")
        
        # Load model and inpaint
        model = get_lama_model(args.checkpoint_dir, args.device, args.fp16)
        
        print("\nRunning inpainting...")
        start_time = time.time()
        result = model.inpaint(image, mask)
        inference_time = time.time() - start_time
        
        # Save result
        if args.output is None:
            stem = Path(args.image).stem
            args.output = f"./feature1/results/{stem}_inpainted.png"
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        print(f"\n{'='*60}")
        print(f"  Results")
        print(f"{'='*60}")
        print(f"  Inference Time: {inference_time:.3f}s")
        print(f"  Output: {args.output}")
        print(f"{'='*60}\n")
        return
    
    # === FLASK SERVER MODE ===
    if args.serve:
        print(f"\n{'='*60}")
        print(f"  LaMa FP16 Flask Server")
        print(f"{'='*60}")
        print(f"  Device: {args.device}")
        print(f"  FP16 Weights: {args.fp16}")
        print(f"  Checkpoint: {args.checkpoint_dir}")
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"{'='*60}\n")
        
        app = create_flask_app(args.checkpoint_dir, args.device, args.fp16)
        app.run(host=args.host, port=args.port, debug=False)
        return
    
    # No mode selected
    parser.print_help()
    print("\nExamples:")
    print("  # Quantize model to FP16")
    print("  python lama_fp16.py --quantize --input_dir ./checkpoints/lama-dilated")
    print("")
    print("  # Run inference")
    print("  python lama_fp16.py --infer --image ./images/test.jpg --mask ./masks/test.png")
    print("")
    print("  # Start Flask server")
    print("  python lama_fp16.py --serve --port 5000")


if __name__ == "__main__":
    main()
