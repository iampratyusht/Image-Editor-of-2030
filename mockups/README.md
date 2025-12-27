# UI Mockups - AI Image Editing Studio <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/figma/figma-original.svg" alt="Figma" width="32" height="32"/>

![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit)
![UI](https://img.shields.io/badge/UI-Mockups-4285F4?logo=figma)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**Visual mockups showcasing the Streamlit web application interface for AI-powered image editing workflows.**

---

## Application Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         APP HOME                                     │
│                    (Feature Selection)                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│      FEATURE 1          │       │      FEATURE 2          │
│   Object Editing        │       │   Neural Relighting     │
│   Pipeline              │       │                         │
└─────────────────────────┘       └─────────────────────────┘
            │                                   │
            ▼                                   ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│  Subfeature Selection   │       │  Adjustment Controls    │
│  • Object Removal       │       │  • Light Direction      │
│  • Object Replacement   │       │  • Intensity Slider     │
│  • Background Fill      │       │  • Z-Depth Control      │
└─────────────────────────┘       └─────────────────────────┘
            │                                   │
            ▼                                   ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│  Flow Selection         │       │  Final Edit & Export    │
│  • Manual Mode          │       │                         │
│  • AI Agent Mode        │       │                         │
└─────────────────────────┘       └─────────────────────────┘
```

---

## Mockup Gallery

### 1. Application Home Screen

The landing page where users select between Feature 1 (Object Editing) and Feature 2 (Neural Relighting).

| App Home |
|----------|
| ![App Home](app_home.png) |

**Components:**
- Feature selection cards
- Project description
- Quick access buttons

---

### 2. Feature 1: Object Editing Pipeline

#### 2.1 Subfeature Options

After selecting Feature 1, users choose the editing operation type.

| Subfeature Selection |
|---------------------|
| ![Subfeature Options](feature1_subfeature_options.png) |

**Available Operations:**
- **Object Removal** - Remove unwanted objects using LaMa inpainting
- **Object Replacement** - Replace objects with AI-generated content (Stable Diffusion + LCM-LoRA)
- **Background Fill** - Replace backgrounds while preserving foreground subjects

---

#### 2.2 Flow Selection (Manual vs Agent)

Users choose between manual object selection or AI Agent mode.

| Flow Selection |
|----------------|
| ![Flow Selection](feature1_subfeature_flow_selection.png) |

**Mode Options:**
- **Manual Mode** - Direct box/point/sketch selection
- **AI Agent Mode** - Natural language interface with Mistral AI

---

#### 2.3 Manual Flow - Selection Methods

When manual mode is selected, users can choose their preferred selection method.

| Manual Selection Options |
|-------------------------|
| ![Manual Options](feature1_selected_subfeature_manual_flow_suboptions.png) |

**Selection Methods:**
- **Box Selection** - Draw bounding box around object
- **Point Selection** - Click on object to segment
- **Sketch Selection** - Draw over the object area
- **Text Selection** - Describe object with text (CLIP-based)

---

#### 2.4 AI Agent Mode

The AI Agent interface for natural language object editing.

| Agent Mode Interface |
|---------------------|
| ![Agent Mode](feature1_selected_subfeature_agent_mockup.png) |

**Features:**
- Natural language input field
- Automatic prompt extraction via Mistral AI
- CLIP prompt generation for segmentation
- Diffusion prompt extraction for replacement/background tasks

---

#### 2.5 Object Description Input

Text-based object description for CLIP segmentation.

| Object Description |
|-------------------|
| ![Object Description](feature1_seleected_subfeature_object_description.png) |

**Workflow:**
1. Upload image
2. Enter object description (e.g., "the red car", "person on left")
3. CLIP + MobileSAM generates segmentation mask
4. Preview mask overlay
5. Apply selected operation

---

### 3. Feature 2: Neural Relighting

#### 3.1 Lighting Adjustment Controls

Interactive controls for adjusting light direction and intensity.

| Relighting Adjustments |
|-----------------------|
| ![Relighting Adjustment](feature2(relighting)_adjustment.png) |

**Controls:**
- **Arrow Drawing** - Draw light direction on canvas
- **Intensity Slider** - Adjust light brightness (0.0 - 3.0)
- **Z-Depth Slider** - Control light depth (-1.0 to 1.0)
- **Light Presets** - Quick buttons for common directions (Left, Right, Top, Bottom)

---

#### 3.2 Final Edit & Export

Preview and export the relit image.

| Final Relighting Output |
|------------------------|
| ![Final Edit](feature2(relighting)_final_edit.png) |

**Output Options:**
- Side-by-side comparison (Original vs Relit)
- Download relit image
- Adjust and re-render

---

## User Journey Summary

### Feature 1: Object Editing

```
Home → Feature 1 → Select Operation → Choose Mode → Select/Describe Object → Preview Mask → Apply → Download
         │              │                  │
         │              ├─ Remove          ├─ Manual (Box/Point/Sketch/Text)
         │              ├─ Replace         └─ Agent (Natural Language)
         │              └─ Background
         │
```

### Feature 2: Neural Relighting

```
Home → Feature 2 → Upload Image → Adjust Light → Preview → Download
                        │              │
                        │              ├─ Direction (Arrow/Presets)
                        │              ├─ Intensity (0-3)
                        │              └─ Z-Depth (-1 to 1)
```

---

## Technical Implementation

These mockups correspond to the Streamlit application defined in `workflows/app.py`:

| Mockup | Code Section |
|--------|--------------|
| `app_home.png` | `main()` function - Feature selection |
| `feature1_subfeature_options.png` | `feature1_page()` - Task type radio buttons |
| `feature1_subfeature_flow_selection.png` | `feature1_page()` - Mode selection |
| `feature1_selected_subfeature_manual_flow_suboptions.png` | `feature1_page()` - Selection method tabs |
| `feature1_selected_subfeature_agent_mockup.png` | `feature1_page()` - Agent text input |
| `feature1_seleected_subfeature_object_description.png` | `feature1_page()` - Text prompt input |
| `feature2(relighting)_adjustment.png` | `feature2_page()` - Light control sliders |
| `feature2(relighting)_final_edit.png` | `feature2_page()` - Result display |

---

## Running the Application

To see these interfaces in action:

```bash
cd workflows
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Related Documentation

- [Main Project README](../workflows/README.md) - Full project setup and usage
- [Feature 1 README](../workflows/feature1/README.md) - Object editing pipeline details
- [Feature 2 README](../workflows/feature2/README.md) - Neural relighting architecture
