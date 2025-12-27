<p align="center">
  <img src="https://img.shields.io/badge/Adobe-Inter%20IIT%20Tech%20Meet%2014.0-FF0000?style=for-the-badge&logo=adobe&logoColor=white" alt="Adobe Inter IIT"/>
</p>

<h1 align="center">2030 AI-Powered Image Editing Workflows</h1>

<p align="center">
  <strong>Reimagining the Future of Creative Expression Through Intelligent Automation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Flutter-3.x-02569B?logo=flutter&logoColor=white" alt="Flutter"/>
  <img src="https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/ONNX-Edge%20Ready-005CED?logo=onnx&logoColor=white" alt="ONNX"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

---

## Overview

This repository presents our solution for the **Adobe Inter IIT Tech Meet 14.0** problem statement: *"Envision the 2030 Editing Workflow."* 

We have developed **production-ready AI pipelines** for interactive image editing that democratize professional-grade creative tools for the next generation of creators. Our solution bridges the gap between complex AI capabilities and intuitive user experiences, enabling seamless object manipulation and intelligent relighting through natural interactions.

### The 2030 Vision

By 2030, image editing will be:
- **Conversational** — Natural language commands replace complex tool menus
- **Intelligent** — AI understands context, intent, and aesthetics
- **Instantaneous** — Real-time processing on mobile and edge devices
- **Accessible** — Professional capabilities available to everyone

Our implementation demonstrates two core workflows that exemplify this vision:

| Workflow | Description | Key Innovation |
|----------|-------------|----------------|
| **Object Editing Pipeline** | Remove, replace, or transform objects using natural selection methods | Multi-modal input (box, click, sketch, text, voice) with AI-powered inpainting |
| **Neural Relighting** | Change lighting direction and intensity with intuitive gestures | Physics-informed neural network with real-time preview |

---

## Deliverables

### A. Product Design (Task 1)

| Deliverable | Status | Link |
|-------------|--------|------|
| 5-7 Screen Mock-ups (Figma) | Complete | [View on Figma](https://www.figma.com/design/Qu8NwfczdxroKhQ1Nh2PMf/Adobe-UIs?node-id=0-1&t=OVnoLUyiDm8tPKiE-1) |
| Short Design Rationale (≤300 words) | Complete | [Design_Rationale.pdf](./Design_Rationale.pdf) |
| UI Mockups Gallery | Complete | [mockups/](./mockups/) |

### B. Editing Ecosystem Analysis (Task 2)

| Deliverable | Status | Link |
|-------------|--------|------|
| 2-Page Market Research Brief | Complete | [Market_research.pdf](./Market_research.pdf) |

### C. Execution (Task 3)

| Deliverable | Status | Link |
|-------------|--------|------|
| Demo Video | Complete | [Demo.mp4](./Demo.mp4) |
| Source Code | Complete | [workflows/](./workflows/) |
| Documentation | Complete | [Documentation.pdf](./Documentation.pdf) |
| Model/Dataset Details | Complete | [workflows/README.md](./workflows/README.md) |

> **Note:** The **Demo.mp4** video contains the actual AI model outputs running locally with full model inference. To run the application yourself, follow the setup instructions in [workflows/README.md](./workflows/README.md)

### D. Optional Creative Artefacts

| Deliverable | Status | Link |
|-------------|--------|------|
| Decision Logs | Complete | [decision_logs.pdf](./decision_logs.pdf) |
| User Personas | Complete | [Design_Rationale.pdf](./Design_Rationale.pdf) |
| Prototype Journeys | Complete | [mockups/README.md](./mockups/README.md) |
| Flutter Mobile App | Complete | [app/](./app/) |

---

## Repository Structure

```
adobe/
├── README.md                      # This file - Project overview and deliverables
├── Demo.mp4                       # Video demonstration of both workflows
├── Design_Rationale.pdf           # Design decisions and rationale (≤300 words)
├── Market_research.pdf            # 2-page editing ecosystem analysis
├── Documentation.pdf              # Technical documentation
├── decision_logs.pdf              # Development decision logs
├── packages.txt                   # System-level dependencies
│
├── mockups/                       # UI/UX Design Mockups
│   ├── README.md                  # Mockup gallery with user flow documentation
│   ├── app_home.png               # Application home screen
│   ├── feature1_subfeature_options.png
│   ├── feature1_subfeature_flow_selection.png
│   ├── feature1_selected_subfeature_manual_flow_suboptions.png
│   ├── feature1_selected_subfeature_agent_mockup.png
│   ├── feature1_seleected_subfeature_object_description.png
│   ├── feature2(relighting)_adjustment.png
│   └── feature2(relighting)_final_edit.png
│
├── workflows/                     # Core AI Pipelines (Streamlit Application)
│   ├── README.md                  # Detailed technical documentation
│   ├── app.py                     # Streamlit web application (1332 lines)
│   ├── requirements.txt           # Python dependencies
│   │
│   ├── feature1/                  # Object Editing Pipeline
│   │   ├── README.md              # Feature 1 documentation
│   │   ├── src/                   # Source code modules
│   │   │   ├── interactive_pipeline.py    # Main interactive GUI
│   │   │   ├── object_removing.py         # Object removal (MobileSAM + LaMa)
│   │   │   ├── object_replacing.py        # Object replacement pipeline
│   │   │   ├── object_replacing_sd.py     # Stable Diffusion replacement
│   │   │   ├── background_filling.py      # Background replacement
│   │   │   ├── inpaint_by_lama.py         # LaMa inpainting module
│   │   │   ├── inpaint_by_sd.py           # SD inpainting with LCM-LoRA
│   │   │   ├── clip_masking.py            # CLIP text-to-mask
│   │   │   ├── masking.py                 # SAM mask utilities
│   │   │   ├── mobilesam.py               # MobileSAM architecture
│   │   │   ├── mobilesamsegment.py        # SAM predictor
│   │   │   ├── agent_decision.py          # Mistral AI agent for NLP
│   │   │   ├── export_sam_onnx.py         # ONNX export for edge
│   │   │   └── lama_fp16.py               # FP16 quantization
│   │   ├── lama/                  # LaMa inpainting source
│   │   ├── utils/                 # Utility functions
│   │   ├── results/               # Sample outputs
│   │   │   ├── object_removing/   # Removal examples
│   │   │   ├── object_replacing/  # Replacement examples
│   │   │   └── background_filling/ # Background examples
│   │   └── LICENSES/              # Third-party licenses
│   │
│   ├── feature2/                  # Neural Relighting Pipeline
│   │   ├── README.md              # Feature 2 documentation
│   │   ├── src/                   # Source code modules
│   │   │   ├── model.py           # TinyRelightNet + FiLM architecture
│   │   │   ├── inference.py       # CLI inference script
│   │   │   ├── interactive_inference.py  # Interactive GUI
│   │   │   ├── preprocess.py      # MiDaS depth, normals, physics
│   │   │   └── postprocess.py     # Output utilities
│   │   ├── notebooks/             # Jupyter notebooks
│   │   ├── results/               # Sample outputs
│   │   └── LICENSES/              # Third-party licenses
│   │
│   └── images/                    # Test images for demo
│
└── app/                           # Flutter Mobile Application
    └── inter_iit_adobe/           # Cross-platform mobile app
        ├── lib/                   # Dart source code
        ├── android/               # Android platform files
        ├── ios/                   # iOS platform files
        ├── web/                   # Web platform files
        ├── macos/                 # macOS platform files
        ├── linux/                 # Linux platform files
        ├── windows/               # Windows platform files
        └── pubspec.yaml           # Flutter dependencies
```

---

## Technical Highlights

### Feature 1: Object Editing Pipeline

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| Segmentation | MobileSAM (ViT-Tiny) | 9.7M | Real-time object masks |
| Text Matching | CLIP ViT-B/32 | 151M | Zero-shot text-to-region |
| Inpainting | LaMa (Dilated) | 51M | Seamless object removal |
| Generation | SD v1.5 + LCM-LoRA | 860M | Fast 4-step replacement |
| Agent | Mistral AI | Cloud | Natural language understanding |

**Key Features:**
- Multi-modal selection: Box, Point, Sketch, Text, or Natural Language
- AI Agent mode with Mistral for conversational editing
- LCM-LoRA enables 4-8 step diffusion (vs. 50 steps standard)
- ONNX export for edge/mobile deployment

### Feature 2: Neural Relighting

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| Depth Estimation | MiDaS Small | 21M | Monocular depth maps |
| Relighting | TinyRelightNet | 1.2M | Physics-aware relight |
| Conditioning | FiLM Layers | — | Dynamic light control |

**Key Features:**
- Arrow-based light direction input
- Physics-based priors (Lambertian + Blinn-Phong shading)
- Real-time preview with adjustable intensity and Z-depth
- Lightweight architecture suitable for mobile

---

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/team24-stack/adobe.git
cd adobe/workflows

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download model weights from Google Drive (see workflows/README.md)

# Run the application
streamlit run app.py
```

For detailed setup instructions including model weight downloads, see [workflows/README.md](./workflows/README.md).

---

## Results

### Object Editing Pipeline

#### Object Removal

| Original | Mask | Result |
|----------|------|--------|
| ![Original](workflows/feature1/results/object_removing/original.jpg) | ![Mask](workflows/feature1/results/object_removing/mask.jpg) | ![Result](workflows/feature1/results/object_removing/output.jpg) |

#### Object Replacing

| Original | Mask | Result |
|----------|------|--------|
| ![Original](workflows/feature1/results/object_replacing/original.jpg) | ![Mask](workflows/feature1/results/object_replacing/mask.jpg) | ![Result](workflows/feature1/results/object_replacing/output.jpg) |

#### Background Filling

| Original | Mask | Result |
|----------|------|--------|
| ![Original](workflows/feature1/results/background_filling/original.jpg) | ![Mask](workflows/feature1/results/background_filling/mask.jpg) | ![Result](workflows/feature1/results/background_filling/output.jpg) |

### Neural Relighting

| Original | Relighted (Right) | Relighted (Left) |
|----------|-------------------|------------------|
| ![Original](workflows/feature2/results/original.jpg) | ![Right](workflows/feature2/results/relighted_from_right_arrow.jpg) | ![Left](workflows/feature2/results/relighted_from_left_arrow.jpg) |

---

## Team

**Team 24**

Inter IIT Tech Meet 14.0 | Adobe Problem Statement

---

## License

This project is licensed under the MIT License. See individual component licenses in `workflows/feature1/LICENSES/` and `workflows/feature2/LICENSES/`.

---

<p align="center">
  <sub>Built with ❤️ for Adobe Inter IIT Tech Meet 14.0</sub>
</p>
