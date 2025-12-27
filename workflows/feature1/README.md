# SegmentAnything + LAMA + Stable Diffusion Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![ONNX](https://img.shields.io/badge/ONNX-Export-005CED?logo=onnx)
![Mobile](https://img.shields.io/badge/Mobile-Optimized-FF6F00?logo=android)
![FP16](https://img.shields.io/badge/FP16-Quantized-9C27B0)

**A production-ready pipeline for interactive image editing optimized for edge devices: remove, replace, or transform objects using state-of-the-art segmentation and inpainting models with ONNX export and quantization support.**

> **Note:** For environment setup and installation instructions, see the main [README.md](../README.md) in the project root.

---

## Features

- **Multi-Modal Segmentation** - Box, Point, Sketch, or Text-based object selection via MobileSAM
- **Object Removal** - Seamless object removal using LaMa (Large Mask) inpainting
- **Object Replacement** - Replace objects with AI-generated content via Stable Diffusion + LCM-LoRA
- **Background Replacement** - Change backgrounds while preserving foreground subjects
- **Configurable Diffusion** - Adjustable diffusion steps (4-25) and mask dilation (1-30)
- **AI Agent Mode** - Natural language interface powered by Mistral AI for automatic prompt extraction
- **Fast Inference** - LCM-LoRA enables 4-8 step diffusion (vs. 50 steps standard)
- **ONNX Export** - Production deployment with FP16/FP32/INT8 ONNX models
- **Edge Optimized** - Lightweight models and fast schedulers for mobile/edge deployment
- **Quantization Support** - FP16 and INT8 quantized models for reduced latency
- **Interactive GUI** - Streamlit web interface with real-time editing

---

## Project Structure

```
feature1/
|-- README.md                  # This documentation
|-- lama/                      # LaMa source code
|-- utils/                     # Utility functions
|   |-- __init__.py
|   |-- amg.py                 # Automatic mask generation utilities
|   |-- crop_for_replacing.py  # Crop utilities for SD inpainting
|   |-- mask_processing.py     # Mask processing utilities
|   +-- utils.py               # General utilities
|-- src/                       # Main source code
|   |-- __init__.py
|   |-- interactive_pipeline.py    # Main interactive GUI application
|   |-- object_removing.py         # Object removal with MobileSAM + LAMA
|   |-- object_replacing.py        # Object replacement pipeline
|   |-- object_replacing_sd.py     # Stable Diffusion replacement
|   |-- background_filling.py      # Background replacement pipeline
|   |-- inpaint_by_lama.py         # LaMa inpainting module
|   |-- inpaint_by_sd.py           # Stable Diffusion inpainting with LCM
|   |-- clip_masking.py            # CLIP-based text-to-mask segmentation
|   |-- masking.py                 # SAM mask prediction utilities
|   |-- mobilesam.py               # MobileSAM model architecture
|   |-- mobilesamsegment.py        # SAM predictor and mask generator
|   |-- agent_decision.py          # LLM agent for natural language processing
|   |-- config.py                  # API keys configuration
|   |-- export_sam_onnx.py         # ONNX export script for MobileSAM
|   |-- lama_fp16.py               # FP16 LAMA utilities
|   +-- object_removing_onnx.py    # ONNX-based object removal (edge)
|-- checkpoints/               # Model weights
|   |-- mobile_sam.pt          # MobileSAM weights
|   |-- lama-dilated/          # LaMa inpainting model
|   |-- stable-diffusion-inpainting/  # SD v1.5 inpainting
|   +-- lcm-lora-sdv1-5/       # LCM-LoRA for fast inference
|-- onnx_export/               # Exported ONNX models for edge deployment
|   |-- lama_fp16.onnx         # LaMa FP16 quantized (smaller, faster)
|   |-- lama_fp32.onnx         # LaMa FP32 (full precision)
|   |-- sam_image_encoder.onnx # SAM image encoder for mobile
|   |-- sam_mask_decoder_point.onnx  # Point-prompt decoder
|   +-- sam_mask_decoder_box.onnx    # Box-prompt decoder
|-- results/                   # Output directory
+-- LICENSES/                  # Third-party model licenses
```

---

## Quick Start

### Interactive Pipeline (Recommended)

```bash
python -m feature1.src.interactive_pipeline --image ./images/test.jpg
```

**Controls:**
| Key | Action |
|-----|--------|
| `1` | Object Removal (LAMA) |
| `2` | Object Replacement (SD+LCM) |
| `3` | Background Replacement (SD+LCM) |
| `b` | Box selection mode |
| `c` | Click/Point selection mode |
| `s` | Sketch/Draw selection mode |
| `a` | Apply operation |
| `z` | Undo |
| `r` | Reset |
| `q` | Quit |

### Object Removal Only

```bash
python -m feature1.src.object_removing --input_img ./images/test.jpg --output_dir ./results
```

### Programmatic Usage

```python
from interactive_pipeline import InteractivePipeline

# Initialize pipeline
pipeline = InteractivePipeline(
    sam_checkpoint="./checkpoints/mobile_sam.pt",
    lama_checkpoint="./checkpoints/lama-dilated",
    device="cpu"
)

# Preload models
pipeline.preload_models(['sam', 'lama', 'sd'])

# Run interactive editing
pipeline.run("./images/input.jpg", output_dir="./results")
```

---

## Architecture

```
+---------------------------------------------------------------------+
|                        User Input                                    |
|              (Image + Selection/Text Prompt)                         |
+---------------------------------------------------------------------+
                                 |
                +----------------+----------------+
                v                                 v
        +---------------+                +---------------+
        |  Manual Mode  |                |  Agent Mode   |
        | Box/Click/    |                | (LLM-powered) |
        | Sketch        |                |               |
        +-------+-------+                +-------+-------+
                |                                 |
                +----------------+----------------+
                                 v
                    +---------------------+
                    |     MobileSAM       |
                    |  (Segmentation)     |
                    |  + CLIP (for text)  |
                    +----------+----------+
                               |
              +----------------+----------------+
              v                v                v
      +-------------+  +-------------+  +-------------+
      |   REMOVE    |  |   REPLACE   |  | BACKGROUND  |
      |   (LAMA)    |  |   (SD+LCM)  |  |   (SD+LCM)  |
      +------+------+  +------+------+  +------+------+
              |                |                |
              +----------------+----------------+
                               |
                               v
                    +---------------------+
                    |    Output Image     |
                    +---------------------+
```

### Data Flow

1. **Input**: User provides image + selection method (box, click, sketch, or text)
2. **Segmentation**: MobileSAM generates object mask from user selection
   - Text mode uses CLIP for zero-shot object detection
3. **Task Routing**: 
   - Remove -> LAMA inpainting (seamless removal)
   - Replace -> SD Inpainting with LCM (4-8 steps)
   - Background -> Inverted mask + SD Inpainting
4. **Output**: Edited image with changes applied

---

## Configuration

### Key Parameters

```python
# Configurable via Streamlit UI sliders
SD_INFERENCE_STEPS = 4-25      # Diffusion steps (default: 8 for replace, 20 for background)
DILATION_KERNEL_SIZE = 1-30    # Mask dilation (default: 12 for replace, 3 for background)
SD_GUIDANCE_SCALE = 1.5-7.5    # Prompt adherence (lower for LCM)
```

### Environment Variables

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0     # Disable HF transfer
export MISTRAL_API_KEY=your_key        # For Agent mode (optional)
```

---

## Model Details

| Component | Model | Parameters | Purpose | Edge-Ready |
|-----------|-------|------------|---------|------------|
| Segmentation | MobileSAM (ViT-Tiny) | 9.7M | Fast object segmentation | Yes (ONNX) |
| Text Matching | CLIP ViT-B/32 | 151M | Text-to-region matching | Yes (ONNX) |
| Inpainting | LaMa (Dilated) | 51M | Object removal | Yes (FP16/INT8) |
| Generation | SD v1.5 Inpainting | 860M | Object/background replacement | LCM optimized |
| Acceleration | LCM-LoRA + PEFT | 67M | 4-25 step inference | Yes |

### Stable Diffusion Optimization for Edge

For mobile deployment, the Stable Diffusion pipeline uses:

- **LCM-LoRA**: Distilled model enabling 4-8 inference steps (vs 50 standard)
- **FP16 Precision**: Half-precision floating point for 2x memory reduction
- **LCMScheduler**: Optimized noise scheduler for few-step inference
- **Guidance Scale**: Lower guidance (1.0-4.0) for faster convergence

```python
# SD Inpainting with LCM optimization
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,  # FP16 for memory efficiency
    safety_checker=None,
)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Fast inference: 4-8 steps instead of 50
result = pipe(prompt, image, mask, num_inference_steps=4, guidance_scale=1.0)
```

---

## ONNX Export and Edge Deployment

This project is optimized for **mobile phones and edge devices** with:

### Exported ONNX Models

| Model | Format | Size | Use Case |
|-------|--------|------|----------|
| `sam_image_encoder.onnx` | FP32 | ~38 MB | SAM image embedding |
| `sam_mask_decoder_point.onnx` | FP32 | ~16 MB | Point-based segmentation |
| `sam_mask_decoder_box.onnx` | FP32 | ~16 MB | Box-based segmentation |
| `lama_fp32.onnx` | FP32 | ~204 MB | High-quality inpainting |
| `lama_fp16.onnx` | FP16 | ~102 MB | Fast mobile inpainting |

### Export Commands

```bash
# Export MobileSAM to ONNX (FP32)
python -m feature1.src.export_sam_onnx --checkpoint ./checkpoints/mobile_sam.pt --output_dir ./onnx_export

# Export with FP16 quantization
python -m feature1.src.export_sam_onnx --checkpoint ./checkpoints/mobile_sam.pt --output_dir ./onnx_export --fp16

# Export with INT8 quantization (smallest size)
python -m feature1.src.export_sam_onnx --checkpoint ./checkpoints/mobile_sam.pt --output_dir ./onnx_export --quantize
```

### Quantization Options

| Precision | Memory | Speed | Accuracy | Best For |
|-----------|--------|-------|----------|----------|
| FP32 | 100% | 1x | 100% | Desktop/Server |
| FP16 | 50% | ~1.5x | ~99.9% | Mobile GPU |
| INT8 | 25% | ~2x | ~99% | Mobile CPU/NPU |

---

## API Reference

### Core Classes

```python
class InteractivePipeline:
    """
    Interactive image editing pipeline supporting:
    - Object removal with LAMA
    - Object replacement with SD + LCM
    - Background fill with SD + LCM
    
    Args:
        sam_checkpoint: Path to MobileSAM weights
        lama_config: Path to LAMA config YAML
        lama_checkpoint: Path to LAMA checkpoint directory
        device: 'cuda' or 'cpu'
    """
    
    def preload_models(self, tasks: List[str] = ['sam', 'lama']):
        """Preload models at startup."""
        
    def run(self, image_path: str, output_dir: str = "./results"):
        """Run interactive editing session."""
```

### Inpainting Functions

```python
def inpaint_img_with_builded_lama(model, img, mask, device="cuda"):
    """
    Inpaint image using pre-loaded LAMA model.
    
    Args:
        model: Pre-loaded LAMA model
        img: Input image (RGB numpy array)
        mask: Binary mask (255 = area to inpaint)
        device: Computation device
    
    Returns:
        Inpainted image (RGB numpy array)
    """

def replace_img_with_sd(img, mask, text_prompt, step=8, device="cuda"):
    """
    Replace masked region with SD-generated content.
    
    Args:
        img: Input image (RGB numpy array)
        mask: Binary mask (255 = area to replace)
        text_prompt: Description of replacement content
        step: LCM inference steps (default: 8)
    
    Returns:
        Image with replaced content
    """
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{segment_inpaint_pipeline,
  title = {SegmentAnything + LAMA + Stable Diffusion Pipeline},
  author = {Team24-Stack},
  year = {2024},
  url = {https://github.com/team24-stack/adobe},
  note = {Interactive image editing with MobileSAM, LAMA, and Stable Diffusion}
}
```

### Related Works

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and others},
  journal={ICCV},
  year={2023}
}

@article{suvorov2022resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and others},
  journal={WACV},
  year={2022}
}

@article{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and others},
  journal={CVPR},
  year={2022}
}
```

---

## License

This project is licensed under the MIT License. See individual model repositories for their respective licenses:
- MobileSAM: Apache 2.0
- LaMa: Apache 2.0
- Stable Diffusion: CreativeML Open RAIL-M
- LCM-LoRA: MIT
- OpenCLIP: MIT

All third-party model licenses are included in the `LICENSES/` directory.

---

## Acknowledgments

- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [LaMa Inpainting](https://github.com/advimman/lama)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [LCM-LoRA](https://github.com/luosiallen/latent-consistency-model)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
