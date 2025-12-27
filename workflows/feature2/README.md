# TinyRelightNet - Neural Image Relighting

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![MiDaS](https://img.shields.io/badge/Depth-MiDaS-orange)
![Dataset](https://img.shields.io/badge/Dataset-VIDIT-blueviolet)

**Real-time neural relighting of images using physics-based priors and a lightweight U-Net with FiLM conditioning - controllable via mouse swipe gestures or arrow drawing.**

> **Note:** For environment setup and installation instructions, see the main [README.md](../README.md) in the project root.

---

## Features

- **Neural Relighting** - Change lighting direction and intensity on any image
- **Lightweight Architecture** - TinyRelightNet U-Net with only ~1.2M parameters
- **FiLM Conditioning** - Feature-wise Linear Modulation for dynamic light control
- **Physics-Based Priors** - Lambertian + Blinn-Phong shading as network input
- **MiDaS Depth Estimation** - Automatic depth and surface normal computation
- **Interactive GUI** - Arrow-based light direction control in Streamlit
- **Adjustable Parameters** - Sliders for intensity (0-3) and Z-depth (-1 to 1)
- **Light Presets** - Quick presets for common lighting directions

---

## Project Structure

```
feature2/
|-- README.md                  # This file
|-- run_inference.sh           # Bash script for batch inference
|-- src/
|   |-- __init__.py
|   |-- model.py               # TinyRelightNet architecture + FiLM
|   |-- inference.py           # CLI inference script
|   |-- interactive_inference.py  # Interactive GUI with mouse swipe
|   |-- preprocess.py          # MiDaS depth, normals, physics relight
|   +-- postprocess.py         # Output saving utilities
|-- weights/
|   +-- model.pth              # Trained model weights
+-- LICENSES/
    |-- LICENSE-MiDaS.txt      # MiDaS model license
    +-- LICENSE-VIDIT.txt      # VIDIT dataset license
```

---

## Quick Start

### Interactive Mode (Recommended)

```bash
# Run interactive relighting with mouse control
python -m feature2.src.interactive_inference \
    --image ./images/portrait.jpg \
    --weights ./feature2/weights/model.pth \
    --device cpu
```

**Controls:**
| Action | Description |
|--------|-------------|
| **Draw Arrow** | Set light direction (start to end) |
| **Intensity Slider** | Adjust light intensity (0.0 - 3.0) |
| **Z Depth Slider** | Adjust light Z coordinate (-1.0 to 1.0) |
| **Light Presets** | Quick buttons for common directions |
| **Apply Button** | Run full model inference |

### CLI Batch Inference

```bash
# Single image
python -m feature2.src.inference \
    --image ./images/test.jpg \
    --weights ./feature2/weights/model.pth \
    --intensity 1.5 \
    --light-dir 1 0 -1 \
    --out-dir ./artifacts

# Process directory
python -m feature2.src.inference \
    --input-dir ./images/ \
    --weights ./feature2/weights/model.pth \
    --out-dir ./artifacts
```

### Programmatic Usage

```python
from feature2.src.model import load_model
from feature2.src.preprocess import load_midas, get_depth_from_midas_bgr
from feature2.src.inference import run_relight_pipeline

# Load models
model = load_model('./feature2/weights/model.pth', device='cpu')
midas, transforms = load_midas('cpu')

# Run relighting
run_relight_pipeline(
    model=model,
    img_path='./images/portrait.jpg',
    device='cpu',
    out_path='./output.png',
    midas=midas,
    midas_transforms=transforms,
    light_vec=[1.0, 0.0, -1.0],  # Light from right
    intensity=1.5
)
```

---

## Architecture

### TinyRelightNet (U-Net + FiLM)

```
+------------------------------------------------------------------+
|                         INPUT (10 channels)                       |
|   RGB(3) + Physics(3) + Normals(3) + Depth(1) = 10 channels      |
+------------------------------------------------------------------+
                                |
                    +-----------v-----------+
                    |      ENCODER          |
                    |-----------------------|
                    |  enc1: 10->32  (H)    |
                    |  enc2: 32->64  (H/2)  | <-- FiLM conditioning
                    |  enc3: 64->128 (H/4)  |
                    |  bott: 128    (H/8)   | <-- FiLM conditioning
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |      DECODER          |
                    |-----------------------|
                    |  dec3: 192->64 (H/4)  |  Skip connections
                    |  dec2: 96->32  (H/2)  |  from encoder
                    |  dec1: 64->32  (H)    |
                    |  out:  32->3   (H)    |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   OUTPUT (3 channels)  |
                    |   Residual + Physics   |
                    |   -> Relit RGB Image   |
                    +------------------------+

Light Conditioning: [lx, ly, lz, intensity] -> FiLM -> gamma, beta modulation
```

### Data Flow

1. **Input Image** -> MiDaS -> **Depth Map** -> Bilateral Smoothing
2. **Depth Map** -> Sobel Gradients -> **Surface Normals**
3. **Normals + Light Vector** -> Lambertian + Blinn-Phong -> **Physics Render**
4. **[RGB, Physics, Normals, Depth]** -> TinyRelightNet -> **Residual**
5. **Physics + Residual** -> Clamp -> **Final Relit Image**

---

## Model Details

### TinyRelightNet Architecture

```python
class TinyRelightNet(nn.Module):
    """
    Lightweight U-Net with FiLM conditioning for neural relighting.
    
    Args:
        in_ch (int): Input channels. Default: 10
            - RGB input: 3
            - Physics render: 3  
            - Surface normals: 3
            - Depth map: 1
        base_ch (int): Base channel count. Default: 32
    
    Input:
        x: Tensor of shape (B, 10, H, W)
        light_cond: Tensor of shape (B, 4) - [lx, ly, lz, intensity]
    
    Output:
        Tensor of shape (B, 3, H, W) - Relit RGB image in [0, 1]
    """
```

### FiLM (Feature-wise Linear Modulation)

```python
class FiLM(nn.Module):
    """
    Applies feature-wise affine transformation conditioned on light parameters.
    
    Given features x and conditioning vector c:
        output = x * (1 + gamma(c)) + beta(c)
    
    Args:
        feat_dim: Feature dimension to modulate
        cond_dim: Conditioning vector dimension (default: 4 for lx,ly,lz,intensity)
    """
```

### Model Statistics

| Property | Value |
|----------|-------|
| Parameters | ~1.2M |
| Input Size | (B, 10, H, W) |
| Output Size | (B, 3, H, W) |
| FiLM Layers | 2 (bottleneck + enc2) |
| Base Channels | 32 |

---

## Training Dataset - VIDIT

This model was trained on the **VIDIT (Virtual Image Dataset for Illumination Transfer)** dataset.

| Property | Details |
|----------|--------|
| **Dataset** | [Nahrawy/VIDIT-Depth-ControlNet](https://huggingface.co/datasets/Nahrawy/VIDIT-Depth-ControlNet) |
| **Original VIDIT** | [VIDIT Dataset](https://github.com/majedelhelou/VIDIT) |
| **Features** | `scene`, `image`, `depth_map`, `direction`, `temperature`, `caption` |
| **Light Directions** | N, S, E, W, NE, NW, SE, SW |
| **License** | CC BY-NC-SA 4.0 |

### Loading the Dataset

```python
from datasets import load_dataset

# Load VIDIT dataset from Hugging Face
hf_ds = load_dataset("Nahrawy/VIDIT-Depth-ControlNet")
train_data = hf_ds["train"]

# Access a sample
sample = train_data[0]
image = sample["image"]           # PIL Image
depth = sample["depth_map"]       # PIL Image (depth)
direction = sample["direction"]   # e.g., "SE", "NW"
temperature = sample["temperature"]
```

### Direction Mapping

```python
# Convert direction string to 3D light vector
DIR2VEC = {
    "N":  ( 0.0, -1.0),  "S":  ( 0.0,  1.0),
    "E":  ( 1.0,  0.0),  "W":  (-1.0,  0.0),
    "NE": ( 1.0, -1.0),  "NW": (-1.0, -1.0),
    "SE": ( 1.0,  1.0),  "SW": (-1.0,  1.0),
}
```

---

## API Reference

### Preprocessing Functions

```python
def load_midas(device: str) -> Tuple[nn.Module, Callable]:
    """Load MiDaS depth estimation model from torch.hub."""

def get_depth_from_midas_bgr(image_bgr, midas, transforms, device) -> np.ndarray:
    """Get normalized depth map [0,1] from BGR image."""

def depth_to_normals_intrinsic(depth, fx=None, fy=None) -> np.ndarray:
    """Convert depth map to surface normals (H,W,3)."""

def physics_relight(input_rgb_srgb, depth, normals, light_vec, intensity) -> np.ndarray:
    """Physics-based relighting with Lambertian + Blinn-Phong shading."""

def swipe_to_direction(start_pt, end_pt, image_shape, z_guess) -> np.ndarray:
    """Convert mouse swipe gesture to 3D light direction vector."""
```

### Model Functions

```python
def load_model(weights_path: str, device: str = 'cpu') -> TinyRelightNet:
    """
    Load TinyRelightNet model from checkpoint.
    
    Supports multiple checkpoint formats:
    - Raw state_dict
    - {'state_dict': ...}
    - {'model_state_dict': ...}
    - DataParallel 'module.' prefix
    """
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{tinyrelightnet2024,
  title = {TinyRelightNet: Lightweight Neural Image Relighting},
  author = {Team24-Stack},
  year = {2024},
  url = {https://github.com/team24-stack/adobe},
  note = {U-Net with FiLM conditioning for real-time relighting}
}
```

### Related Works

```bibtex
@article{ranftl2020midas,
  title={Towards Robust Monocular Depth Estimation},
  author={Ranftl, Rene and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladlen},
  journal={IEEE TPAMI},
  year={2022}
}

@inproceedings{perez2018film,
  title={FiLM: Visual Reasoning with a General Conditioning Layer},
  author={Perez, Ethan and Strub, Florian and De Vries, Harm and Dumoulin, Vincent and Courville, Aaron},
  booktitle={AAAI},
  year={2018}
}

@inproceedings{helou2020vidit,
  title={VIDIT: Virtual Image Dataset for Illumination Transfer},
  author={El Helou, Majed and Zhou, Ruofan and Barthas, Johan and Susstrunk, Sabine},
  booktitle={arXiv preprint arXiv:2005.05460},
  year={2020}
}
```

---

## License

This project is licensed under the MIT License.

Third-party model and dataset licenses are included in the `LICENSES/` directory:
- **MiDaS**: MIT License (Intel ISL)
- **VIDIT Dataset**: CC BY-NC-SA 4.0 (EPFL)

---

## Acknowledgments

- [MiDaS](https://github.com/isl-org/MiDaS) - Intel ISL depth estimation
- [VIDIT Dataset](https://github.com/majedelhelou/VIDIT) - Virtual Image Dataset for Illumination Transfer (EPFL)
- [VIDIT on Hugging Face](https://huggingface.co/datasets/Nahrawy/VIDIT-Depth-ControlNet) - HF-hosted version with depth maps
- [FiLM](https://arxiv.org/abs/1709.07871) - Feature-wise Linear Modulation
- PyTorch team for the deep learning framework
