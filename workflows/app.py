"""
Streamlit Demo Application for Image Editing Workflows

Features:
1. Object Removal/Replacement/Background Fill (MobileSAM + LAMA + Stable Diffusion)
2. Neural Relighting (TinyRelightNet + MiDaS)

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import torch
import os
import sys
import tempfile
from PIL import Image
from pathlib import Path
import time

# Disable hf_transfer to avoid download errors
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Check canvas availability - may fail with newer Streamlit versions
CANVAS_AVAILABLE = False
try:
    from streamlit_drawable_canvas import st_canvas
    # Test if the canvas actually works with current Streamlit version
    import streamlit.elements.image as st_image
    if hasattr(st_image, 'image_to_url'):
        CANVAS_AVAILABLE = True
except (ImportError, AttributeError):
    CANVAS_AVAILABLE = False

# Add feature1 to path for lama imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'feature1'))

# Page config
st.set_page_config(
    page_title="AI Image Editing Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'feature1_models_loaded' not in st.session_state:
        st.session_state.feature1_models_loaded = False
    if 'feature2_models_loaded' not in st.session_state:
        st.session_state.feature2_models_loaded = False
    if 'sam_predictor' not in st.session_state:
        st.session_state.sam_predictor = None
    if 'lama_model' not in st.session_state:
        st.session_state.lama_model = None
    if 'relight_model' not in st.session_state:
        st.session_state.relight_model = None
    if 'midas_model' not in st.session_state:
        st.session_state.midas_model = None
    if 'current_mask' not in st.session_state:
        st.session_state.current_mask = None
    if 'result_image' not in st.session_state:
        st.session_state.result_image = None
    if 'models_preloaded' not in st.session_state:
        st.session_state.models_preloaded = False
    if 'agent_diffusion_prompt' not in st.session_state:
        st.session_state.agent_diffusion_prompt = None
    if 'agent_negative_prompt' not in st.session_state:
        st.session_state.agent_negative_prompt = None


init_session_state()


# =============================================================================
# DEVICE DETECTION
# =============================================================================

@st.cache_resource
def get_device():
    """Get available device."""
    return "cpu"


# =============================================================================
# CLOUD DEPLOYMENT DETECTION
# =============================================================================

IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') == 'streamlit' or \
                     os.path.exists('/mount/src') or \
                     'STREAMLIT' in os.environ.get('HOME', '')

# =============================================================================
# PRELOAD MODELS ON STARTUP
# =============================================================================

@st.cache_resource(show_spinner=False)
def preload_all_models():
    """Preload all models at startup for faster inference."""
    models = {}
    
    # Load SAM
    try:
        from feature1.src.mobilesamsegment import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_t"](checkpoint="./feature1/checkpoints/mobile_sam.pt")
        sam.to(get_device())
        sam.eval()
        models['sam'] = SamPredictor(sam)
    except Exception as e:
        models['sam'] = None
        print(f"Failed to preload SAM: {e}")
    
    # Load LAMA
    try:
        from feature1.src.inpaint_by_lama import build_lama_model
        models['lama'] = build_lama_model(
            config_p="./feature1/lama/configs/prediction/default.yaml",
            ckpt_p="./feature1/checkpoints/lama-dilated",
            device=get_device()
        )
    except Exception as e:
        models['lama'] = None
        print(f"Failed to preload LAMA: {e}")
    
    # Load Stable Diffusion Inpainting Pipeline
    try:
        from diffusers import StableDiffusionInpaintPipeline, LCMScheduler
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "./feature1/checkpoints/stable-diffusion-inpainting",
            torch_dtype=torch.float32,
            safety_checker=None
        )
        # Load LCM-LoRA for faster inference
        try:
            pipe.load_lora_weights("./feature1/checkpoints/lcm-lora-sdv1-5")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        except Exception as lora_e:
            print(f"LCM-LoRA not loaded (using default scheduler): {lora_e}")
        pipe.to(get_device())
        models['sd_inpaint'] = pipe
    except Exception as e:
        models['sd_inpaint'] = None
        print(f"Failed to preload Stable Diffusion: {e}")
    
    # Check agent availability
    try:
        from feature1.src.agent_decision import check_api_key, create_agent_chain, process_user_prompt
        check_api_key()
        models['agent_chain'] = create_agent_chain()
        models['agent_available'] = True
    except Exception as e:
        models['agent_chain'] = None
        models['agent_available'] = False
        print(f"Agent not available (will use direct CLIP): {e}")
    
    return models


# Preload models immediately when script runs
with st.spinner("🚀 Loading AI models... This may take a moment on first run."):
    PRELOADED_MODELS = preload_all_models()


# =============================================================================
# FEATURE 1: OBJECT EDITING (SAM + LAMA + SD)
# =============================================================================

def load_sam_model():
    """Get preloaded SAM model."""
    return PRELOADED_MODELS.get('sam')


def load_lama_model():
    """Get preloaded LAMA model."""
    return PRELOADED_MODELS.get('lama')


def load_sd_inpaint_model():
    """Get preloaded Stable Diffusion inpainting model."""
    return PRELOADED_MODELS.get('sd_inpaint')


def is_agent_available():
    """Check if agent is available."""
    return PRELOADED_MODELS.get('agent_available', False)


def process_text_with_agent(user_prompt: str, task_type: str) -> dict:
    """
    Process user text with agent to extract prompts.
    Falls back to direct input if agent unavailable.
    
    Returns:
        dict with keys: clip_prompt, diffusion_prompt, negative_prompt
    """
    result = {
        'clip_prompt': user_prompt,
        'diffusion_prompt': user_prompt,
        'negative_prompt': "blurry, low quality, distorted, deformed, ugly"
    }
    
    if is_agent_available():
        try:
            from feature1.src.agent_decision import process_user_prompt
            agent_chain = PRELOADED_MODELS.get('agent_chain')
            if agent_chain:
                decision = process_user_prompt(agent_chain, user_prompt)
                if 'error' not in decision:
                    result['clip_prompt'] = decision.get('clip_prompt', user_prompt)
                    if task_type != 'remove':
                        result['diffusion_prompt'] = decision.get('diffusion_prompt', user_prompt)
                        result['negative_prompt'] = decision.get('negative_diffusion_prompt', result['negative_prompt'])
                    else:
                        result['diffusion_prompt'] = None
                        result['negative_prompt'] = None
        except Exception as e:
            st.warning(f"Agent processing failed, using direct input: {e}")
    
    return result


def generate_mask_from_points(predictor, image, points, labels):
    """Generate mask from point prompts."""
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=True
    )
    best_idx = np.argmax(scores)
    return (masks[best_idx] * 255).astype(np.uint8)


def generate_mask_from_box(predictor, image, box):
    """Generate mask from bounding box."""
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(box),
        multimask_output=False
    )
    return (masks[0] * 255).astype(np.uint8)


def generate_mask_from_text(image, text_prompt, device):
    """Generate mask from text prompt using CLIP."""
    try:
        from feature1.src.clip_masking import find_object_with_text
        mask = find_object_with_text(
            image, text_prompt,
            sam_ckpt="./feature1/checkpoints/mobile_sam.pt",
            model_type="vit_t",
            device=device
        )
        return mask
    except Exception as e:
        st.error(f"Text-based masking failed: {e}")
        return None


def dilate_mask(mask, kernel_size=12):
    """Dilate mask for better edge coverage."""
    if kernel_size <= 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def perform_removal(image, mask, lama_model, device):
    """Remove object using LAMA."""
    from feature1.src.inpaint_by_lama import inpaint_img_with_builded_lama
    dilated = dilate_mask(mask, 12)
    result = inpaint_img_with_builded_lama(lama_model, image, dilated, device=device)
    return result


def perform_replacement(image, mask, prompt, negative_prompt, device, steps=8, dilation=12):
    """Replace object using Stable Diffusion (uses preloaded model if available)."""
    dilated = dilate_mask(mask, dilation)
    
    # Try to use preloaded SD model
    sd_pipe = load_sd_inpaint_model()
    if sd_pipe is not None:
        try:
            from PIL import Image as PILImage
            # Prepare inputs
            img_pil = PILImage.fromarray(image).convert("RGB")
            mask_pil = PILImage.fromarray(dilated).convert("L")
            
            # Run inference
            result = sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "blurry, low quality, distorted",
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=steps,
                guidance_scale=1.5
            ).images[0]
            return np.array(result)
        except Exception as e:
            st.warning(f"Preloaded SD failed, falling back: {e}")
    
    # Fallback to original function
    from feature1.src.inpaint_by_sd import replace_img_with_sd
    result = replace_img_with_sd(
        image, dilated, prompt,
        negative_prompt=negative_prompt or "blurry, low quality, distorted",
        step=steps, device=device
    )
    return result


def perform_background_fill(image, mask, prompt, device, steps=20, dilation=3):
    """Fill background using Stable Diffusion (uses preloaded model if available)."""
    # Invert mask for background fill (mask should cover background, not object)
    inverted_mask = 255 - mask
    
    # Apply dilation to the inverted mask
    if dilation > 0:
        inverted_mask = dilate_mask(inverted_mask, dilation)
    
    # Try to use preloaded SD model
    sd_pipe = load_sd_inpaint_model()
    if sd_pipe is not None:
        try:
            from PIL import Image as PILImage
            # Prepare inputs
            img_pil = PILImage.fromarray(image).convert("RGB")
            mask_pil = PILImage.fromarray(inverted_mask).convert("L")
            
            # Run inference
            result = sd_pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed",
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=steps,
                guidance_scale=7.5
            ).images[0]
            return np.array(result)
        except Exception as e:
            st.warning(f"Preloaded SD failed, falling back: {e}")
    
    # Fallback to original function
    from feature1.src.inpaint_by_sd import fill_background_with_sd
    result = fill_background_with_sd(image, mask, prompt, step=steps, device=device)
    return result


# =============================================================================
# FEATURE 2: NEURAL RELIGHTING
# =============================================================================

@st.cache_resource
def load_relight_model():
    """Load TinyRelightNet model."""
    try:
        from feature2.src.model import load_model
        model = load_model("./feature2/weights/model.pth", device=get_device())
        return model
    except Exception as e:
        st.error(f"Failed to load relighting model: {e}")
        return None


@st.cache_resource
def load_midas():
    """Load MiDaS depth model."""
    try:
        from feature2.src.preprocess import load_midas as load_midas_fn
        midas, transforms = load_midas_fn(get_device())
        return midas, transforms
    except Exception as e:
        st.error(f"Failed to load MiDaS: {e}")
        return None, None


def compute_relighting(image_bgr, model, midas, midas_transforms, light_vec, intensity, device):
    """Compute neural relighting."""
    from feature2.src.preprocess import (
        get_depth_from_midas_bgr, smooth_depth, depth_to_normals_intrinsic,
        physics_relight, make_divisible_hw
    )
    
    H0, W0 = image_bgr.shape[:2]
    H_res, W_res = make_divisible_hw(H0, W0, divisor=8)
    img_resized = cv2.resize(image_bgr, (W_res, H_res), interpolation=cv2.INTER_AREA)
    
    input_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Depth and normals
    depth = get_depth_from_midas_bgr(img_resized, midas, midas_transforms, device)
    depth_s = smooth_depth(depth)
    normals = depth_to_normals_intrinsic(depth_s)
    
    # Normalize light vector
    light_vec = np.array(light_vec, dtype=np.float32)
    light_vec /= (np.linalg.norm(light_vec) + 1e-9)
    
    # Physics-based relight
    physics_img = physics_relight(
        input_rgb_srgb=input_rgb,
        depth=depth_s,
        normals=normals,
        light_vec=light_vec,
        is_point=False,
        intensity=float(intensity),
    )
    
    # Build network input
    inp_chw = np.transpose(input_rgb, (2, 0, 1))
    phys_chw = np.transpose(physics_img, (2, 0, 1))
    normals_chw = np.transpose(normals, (2, 0, 1))
    depth_chw = depth_s[None, ...]
    
    net_in_np = np.concatenate([inp_chw, phys_chw, normals_chw, depth_chw], axis=0)
    net_in = torch.from_numpy(net_in_np).unsqueeze(0).float().to(device)
    
    lx, ly, lz = light_vec.tolist()
    light_cond = torch.tensor([[lx, ly, lz, float(intensity)]], dtype=torch.float32).to(device)
    phys_t = torch.from_numpy(phys_chw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        residual = model(net_in, light_cond)
        out = torch.clamp(phys_t + (residual - 0.5) * 0.6, 0.0, 1.0)
    
    result = out[0].cpu().numpy().transpose(1, 2, 0)
    result = cv2.resize(result, (W0, H0), interpolation=cv2.INTER_LINEAR)
    result_uint8 = (result * 255).astype(np.uint8)
    
    return result_uint8, physics_img, depth_s, normals


# =============================================================================
# UI COMPONENTS
# =============================================================================

def image_to_pil(image_array):
    """Convert numpy array to PIL Image."""
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)


def pil_to_array(pil_image):
    """Convert PIL Image to numpy array."""
    return np.array(pil_image)


def create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """Create image with mask overlay."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Image Editing Studio</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.shields.io/badge/PyTorch-Powered-red?logo=pytorch", width=150)
        st.markdown("---")
        
        feature = st.radio(
            "Select Feature",
            ["🖼️ Object Editing", "💡 Neural Relighting"],
            help="Choose the editing feature you want to use"
        )
        
        st.markdown("---")
        st.markdown("### Device Info")
        device = get_device()
        st.info("💻 Running on CPU (Mobile Optimized)")
        
        st.markdown("---")
        st.markdown("### Model Status")
        
        if IS_STREAMLIT_CLOUD:
            st.warning("☁️ **Cloud Demo Mode**")
            st.caption("Models require local setup. See GitHub for instructions.")
            sam_status = "⚠️ Cloud"
            lama_status = "⚠️ Cloud"
        else:
            sam_status = "✅ Loaded" if PRELOADED_MODELS.get('sam') else "❌ Failed"
            lama_status = "✅ Loaded" if PRELOADED_MODELS.get('lama') else "❌ Failed"
        
        st.markdown(f"- **SAM:** {sam_status}")
        st.markdown(f"- **LAMA:** {lama_status}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Object Editing** uses:
        - MobileSAM for segmentation
        - LAMA for inpainting
        - Stable Diffusion + LCM
        
        **Neural Relighting** uses:
        - MiDaS for depth estimation
        - TinyRelightNet (U-Net + FiLM)
        """)
    
    # Main content
    if feature == "🖼️ Object Editing":
        render_object_editing_page()
    else:
        render_relighting_page()


def render_object_editing_page():
    """Render the object editing feature page."""
    st.header("🖼️ Object Editing Pipeline")
    
    # Cloud demo notice
    if IS_STREAMLIT_CLOUD:
        st.info("""
        ☁️ **Cloud Demo Mode** - This is a UI demonstration. For full functionality with AI models:
        1. Clone the repo: `git clone https://github.com/team24-stack/adobe.git`
        2. Download model weights from Google Drive (see README)
        3. Run locally: `streamlit run app.py`
        
        📹 [Watch Demo Video](https://github.com/team24-stack/adobe) | 📄 [View Documentation](https://github.com/team24-stack/adobe/blob/main/workflows/README.md)
        """)
    
    st.markdown("""
    <div class="info-box">
    <b>Capabilities:</b> Remove objects, replace objects with AI-generated content, 
    or change backgrounds while keeping the subject.
    </div>
    """, unsafe_allow_html=True)
    
    # Task selection
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧹 Remove Object"):
            st.session_state.edit_task = "remove"
    with col2:
        if st.button("🔄 Replace Object"):
            st.session_state.edit_task = "replace"
    with col3:
        if st.button("🌅 Change Background"):
            st.session_state.edit_task = "background"
    
    task = st.session_state.get('edit_task', 'remove')
    st.info(f"Current Task: **{task.upper()}**")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key="object_edit_upload"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Selection method
        st.subheader("Selection Method")
        selection_method = st.radio(
            "Choose how to select the object:",
            ["📝 Text Prompt (AI Agent)", "📦 Bounding Box", "👆 Click Points", "🎨 Sketch Mask (Draw)"],
            horizontal=True
        )
        
        mask = None
        
        if "Text Prompt" in selection_method:
            # Show agent status
            if is_agent_available():
                st.success("🤖 AI Agent is active - will extract segmentation & diffusion prompts automatically")
            else:
                st.info("💡 Using direct CLIP matching (set MISTRAL_API_KEY for AI agent)")
            
            text_prompt = st.text_input(
                "Describe what you want to do:",
                placeholder="e.g., 'remove the dog', 'replace the car with a red sports car', 'change background to beach sunset'"
            )
            
            if st.button("🔍 Process with AI", type="primary"):
                if text_prompt:
                    with st.spinner("Processing with AI..."):
                        # Get the current task
                        task = st.session_state.get('edit_task', 'remove')
                        
                        # Process with agent (or fallback to direct)
                        extracted = process_text_with_agent(text_prompt, task)
                        clip_prompt = extracted['clip_prompt']
                        
                        # Store diffusion prompts for later use
                        st.session_state.agent_diffusion_prompt = extracted['diffusion_prompt']
                        st.session_state.agent_negative_prompt = extracted['negative_prompt']
                        
                        # Show what the agent extracted
                        if is_agent_available():
                            st.info(f"**Segmentation target:** {clip_prompt}")
                            if task != 'remove' and extracted['diffusion_prompt']:
                                st.info(f"**Generation prompt:** {extracted['diffusion_prompt']}")
                        
                        # Generate mask using CLIP
                        mask = generate_mask_from_text(image_np, clip_prompt, get_device())
                        if mask is not None:
                            st.session_state.current_mask = mask
                            st.success("Object found!")
                        else:
                            st.error("Could not find the specified object. Try a different description.")
                else:
                    st.warning("Please enter a text description.")
        
        elif "Bounding Box" in selection_method:
            img_h, img_w = image_np.shape[:2]
            
            # Choose input method
            bbox_method = st.radio(
                "Choose bounding box input method:",
                ["🎨 Draw on Image (Interactive)", "🔢 Enter Coordinates (Manual)"],
                horizontal=True
            )
            
            if "Draw on Image" in bbox_method:
                if CANVAS_AVAILABLE:
                    st.markdown("**Draw a rectangle around the object:**")
                    st.caption("Click and drag to draw a bounding box. Draw only ONE rectangle.")
                    
                    # Calculate canvas size (max 700px width, maintain aspect ratio)
                    canvas_width = min(700, img_w)
                    scale = canvas_width / img_w
                    canvas_height = int(img_h * scale)
                    
                    # Resize image for canvas display
                    canvas_image = cv2.resize(image_np, (canvas_width, canvas_height))
                    
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.2)",
                        stroke_width=2,
                        stroke_color="#FF0000",
                        background_image=Image.fromarray(canvas_image),
                        update_streamlit=True,
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode="rect",
                        key="bbox_canvas",
                    )
                    
                    # Extract bounding box from canvas
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data.get("objects", [])
                        if objects:
                            # Get the last drawn rectangle
                            rect = objects[-1]
                            # Scale coordinates back to original image size
                            x1 = int(rect["left"] / scale)
                            y1 = int(rect["top"] / scale)
                            x2 = int((rect["left"] + rect["width"]) / scale)
                            y2 = int((rect["top"] + rect["height"]) / scale)
                            
                            # Clamp to image bounds
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_w, x2), min(img_h, y2)
                            
                            st.success(f"Box coordinates: ({x1}, {y1}) → ({x2}, {y2})")
                            
                            if st.button("📦 Generate Mask from Drawn Box", type="primary"):
                                with st.spinner("Generating mask with SAM..."):
                                    predictor = load_sam_model()
                                    if predictor:
                                        mask = generate_mask_from_box(predictor, image_np, [x1, y1, x2, y2])
                                        st.session_state.current_mask = mask
                                        st.success("Mask generated!")
                        else:
                            st.info("Draw a rectangle on the image above to select an object.")
                else:
                    st.warning("⚠️ Interactive drawing requires `streamlit-drawable-canvas`. Install it with:")
                    st.code("pip install streamlit-drawable-canvas")
                    st.info("Switching to manual coordinate input...")
                    bbox_method = "Manual"  # Fallback
            
            if "Enter Coordinates" in bbox_method or not CANVAS_AVAILABLE:
                st.markdown("**Enter bounding box coordinates:**")
                col_x1, col_y1, col_x2, col_y2 = st.columns(4)
                with col_x1:
                    x1 = st.number_input("X1", min_value=0, max_value=img_w, value=min(50, img_w // 4))
                with col_y1:
                    y1 = st.number_input("Y1", min_value=0, max_value=img_h, value=min(50, img_h // 4))
                with col_x2:
                    x2 = st.number_input("X2", min_value=0, max_value=img_w, value=min(200, img_w * 3 // 4))
                with col_y2:
                    y2 = st.number_input("Y2", min_value=0, max_value=img_h, value=min(200, img_h * 3 // 4))
                
                if st.button("📦 Generate Mask from Box", type="primary"):
                    with st.spinner("Generating mask with SAM..."):
                        predictor = load_sam_model()
                        if predictor:
                            mask = generate_mask_from_box(predictor, image_np, [x1, y1, x2, y2])
                            st.session_state.current_mask = mask
                            st.success("Mask generated!")
        
        elif "Click Points" in selection_method:
            st.markdown("**Click on the image or enter coordinates:**")
            
            # Choose input method for points too
            points_method = st.radio(
                "Choose point input method:",
                ["🎨 Click on Image (Interactive)", "🔢 Enter Coordinates (Manual)"],
                horizontal=True,
                key="points_method"
            )
            
            if "Click on Image" in points_method and CANVAS_AVAILABLE:
                img_h, img_w = image_np.shape[:2]
                canvas_width = min(700, img_w)
                scale = canvas_width / img_w
                canvas_height = int(img_h * scale)
                
                canvas_image = cv2.resize(image_np, (canvas_width, canvas_height))
                
                st.caption("Click to add foreground points (green). Hold Shift+Click for background points (red).")
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.8)",
                    stroke_width=0,
                    stroke_color="#00FF00",
                    background_image=Image.fromarray(canvas_image),
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="point",
                    point_display_radius=5,
                    key="points_canvas",
                )
                
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    if objects:
                        points = []
                        labels = []
                        for obj in objects:
                            if obj["type"] == "circle":
                                px = int((obj["left"] + obj["radius"]) / scale)
                                py = int((obj["top"] + obj["radius"]) / scale)
                                points.append([px, py])
                                labels.append(1)  # foreground
                        
                        if points:
                            st.success(f"Selected {len(points)} point(s)")
                            
                            if st.button("👆 Generate Mask from Points", type="primary"):
                                with st.spinner("Generating mask with SAM..."):
                                    predictor = load_sam_model()
                                    if predictor:
                                        mask = generate_mask_from_points(predictor, image_np, points, labels)
                                        st.session_state.current_mask = mask
                                        st.success("Mask generated!")
                    else:
                        st.info("Click on the image to add selection points.")
            
            elif "Enter Coordinates" in points_method or not CANVAS_AVAILABLE:
                points_input = st.text_input(
                    "Points (x1,y1;x2,y2;...):",
                    placeholder="100,150;200,180"
                )
                
                if st.button("👆 Generate Mask from Points", type="primary"):
                    if points_input:
                        try:
                            points = []
                            for p in points_input.split(';'):
                                x, y = map(int, p.strip().split(','))
                                points.append([x, y])
                            labels = [1] * len(points)  # All foreground
                            
                            with st.spinner("Generating mask with SAM..."):
                                predictor = load_sam_model()
                                if predictor:
                                    mask = generate_mask_from_points(predictor, image_np, points, labels)
                                    st.session_state.current_mask = mask
                                    st.success("Mask generated!")
                        except Exception as e:
                            st.error(f"Invalid point format: {e}")
        
        elif "Sketch Mask" in selection_method:
            st.markdown("**🎨 Sketch over the object to select it:**")
            
            img_h, img_w = image_np.shape[:2]
            
            if CANVAS_AVAILABLE:
                # Calculate canvas size
                canvas_width = min(700, img_w)
                scale = canvas_width / img_w
                canvas_height = int(img_h * scale)
                
                canvas_image = cv2.resize(image_np, (canvas_width, canvas_height))
                
                st.caption("**Draw freely over the object** - topmost and bottommost points will be extracted to create a bounding box for SAM.")
                
                # Brush settings
                stroke_width = st.slider("Brush Size", 3, 30, 10, key="sketch_stroke_width")
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.5)",
                    stroke_width=stroke_width,
                    stroke_color="#00FF00",
                    background_image=Image.fromarray(canvas_image),
                    update_streamlit=False,  # Don't auto-update to prevent infinite loops
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="freedraw",
                    key="sketch_canvas",
                )
                
                # Initialize session state for sketch box
                if "sketch_box" not in st.session_state:
                    st.session_state.sketch_box = None
                
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    if objects:
                        # Extract all points from freedraw paths
                        all_points = []
                        for obj in objects:
                            if obj["type"] == "path" and "path" in obj:
                                # Extract points from SVG path commands
                                for cmd in obj["path"]:
                                    if len(cmd) >= 3:
                                        # Commands like ["M", x, y], ["L", x, y], ["Q", cx, cy, x, y]
                                        if cmd[0] in ["M", "L"]:
                                            px, py = cmd[1], cmd[2]
                                            all_points.append((px, py))
                                        elif cmd[0] == "Q" and len(cmd) >= 5:
                                            # Quadratic curve - take the end point
                                            px, py = cmd[3], cmd[4]
                                            all_points.append((px, py))
                        
                        if all_points:
                            # Find topmost (min y) and bottommost (max y) points
                            all_points_np = np.array(all_points)
                            
                            # Get bounding box from all drawn points
                            min_x = int(np.min(all_points_np[:, 0]) / scale)
                            max_x = int(np.max(all_points_np[:, 0]) / scale)
                            min_y = int(np.min(all_points_np[:, 1]) / scale)
                            max_y = int(np.max(all_points_np[:, 1]) / scale)
                            
                            # Clamp to image bounds
                            x1 = max(0, min_x)
                            y1 = max(0, min_y)
                            x2 = min(img_w - 1, max_x)
                            y2 = min(img_h - 1, max_y)
                            
                            # Add small padding
                            padding = 5
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(img_w - 1, x2 + padding)
                            y2 = min(img_h - 1, y2 + padding)
                            
                            st.session_state.sketch_box = [x1, y1, x2, y2]
                            st.info(f"📦 Bounding box from sketch: ({x1}, {y1}) to ({x2}, {y2})")
                        else:
                            st.info("Draw over the object to select it.")
                    else:
                        st.info("Draw over the object to select it.")
                
                # Button to generate mask (separate from canvas logic)
                if st.session_state.sketch_box is not None:
                    if st.button("🎯 Generate SAM Mask from Sketch", type="primary", key="sketch_gen_btn"):
                        box = st.session_state.sketch_box
                        with st.spinner("Generating mask with SAM..."):
                            predictor = load_sam_model()
                            if predictor:
                                mask = generate_mask_from_box(predictor, image_np, box)
                                st.session_state.current_mask = mask
                                st.session_state.sketch_box = None  # Clear the box
                                st.success("✅ Mask generated successfully!")
                                st.rerun()
            else:
                # Fallback: Text-based point entry or mask upload
                st.warning("⚠️ Interactive canvas not available with this Streamlit version.")
                st.info("**Alternative: Enter stroke points manually or upload a mask**")
                
                sketch_method = st.radio(
                    "Choose method:",
                    ["📦 Bounding Box (SAM)", "📤 Upload Mask Image"],
                    horizontal=True,
                    key="sketch_fallback_method"
                )
                
                if "Bounding Box" in sketch_method:
                    st.markdown("Enter bounding box coordinates (SAM will segment the object inside):")
                    st.caption(f"Image size: {img_w} x {img_h}")
                    
                    col_box1, col_box2 = st.columns(2)
                    with col_box1:
                        box_x1 = st.number_input("X1 (left)", min_value=0, max_value=img_w-1, value=0, key="box_x1")
                        box_y1 = st.number_input("Y1 (top)", min_value=0, max_value=img_h-1, value=0, key="box_y1")
                    with col_box2:
                        box_x2 = st.number_input("X2 (right)", min_value=0, max_value=img_w-1, value=min(100, img_w-1), key="box_x2")
                        box_y2 = st.number_input("Y2 (bottom)", min_value=0, max_value=img_h-1, value=min(100, img_h-1), key="box_y2")
                    
                    if st.button("🎯 Generate Mask from Box", type="primary", key="gen_box_mask"):
                        if box_x2 > box_x1 and box_y2 > box_y1:
                            with st.spinner("Generating mask from bounding box..."):
                                predictor = load_sam_model()
                                if predictor:
                                    mask = generate_mask_from_box(predictor, image_np, [box_x1, box_y1, box_x2, box_y2])
                                    st.session_state.current_mask = mask
                                    st.success(f"Mask generated from box ({box_x1},{box_y1}) to ({box_x2},{box_y2})!")
                                    st.rerun()
                        else:
                            st.warning("X2 must be greater than X1, and Y2 must be greater than Y1.")
                
                elif "Upload Mask" in sketch_method:
                    st.markdown("Upload a black & white mask image (white = object):")
                    mask_file = st.file_uploader(
                        "Upload mask image",
                        type=['jpg', 'jpeg', 'png', 'bmp'],
                        key="mask_upload"
                    )
                    
                    if mask_file is not None:
                        mask_img = Image.open(mask_file).convert('L')
                        mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
                        mask_np = np.array(mask_img)
                        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                        
                        st.image(binary_mask, caption="Uploaded Mask Preview", use_container_width=True)
                        
                        if st.button("✅ Use This Mask", type="primary", key="use_uploaded_mask"):
                            st.session_state.current_mask = binary_mask.astype(np.uint8)
                            st.success("Mask loaded!")
                            st.rerun()
        
        # Show mask preview
        if st.session_state.current_mask is not None:
            mask = st.session_state.current_mask
            with col2:
                st.subheader("Mask Preview")
                overlay = create_mask_overlay(image_np, mask, color=(255, 0, 0), alpha=0.4)
                st.image(overlay, use_container_width=True)
                
                mask_coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
                st.caption(f"Mask coverage: {mask_coverage:.1f}%")
        
        # Task-specific options
        st.markdown("---")
        st.subheader("Apply Operation")
        
        if task == "remove":
            st.markdown("**Remove the selected object using LAMA inpainting.**")
            dilation = st.slider("Mask Dilation", 0, 30, 12, help="Expand mask for better edge coverage")
            
            if st.button("🧹 Remove Object", type="primary", disabled=(st.session_state.current_mask is None)):
                with st.spinner("Removing object with LAMA..."):
                    lama_model = load_lama_model()
                    if lama_model and st.session_state.current_mask is not None:
                        result = perform_removal(image_np, st.session_state.current_mask, lama_model, get_device())
                        st.session_state.result_image = result
        
        elif task == "replace":
            st.markdown("**Replace the selected object with AI-generated content.**")
            
            # Pre-fill with agent-extracted prompt if available
            default_replace = st.session_state.get('agent_diffusion_prompt', '')
            default_negative = st.session_state.get('agent_negative_prompt', 'blurry, low quality, distorted, deformed, ugly')
            
            if default_replace and is_agent_available():
                st.success(f"🤖 AI suggested: *{default_replace}*")
            
            replace_prompt = st.text_input(
                "What should replace the object?",
                value=default_replace if default_replace else "",
                placeholder="e.g., 'a cute cat sitting', 'a red sports car'"
            )
            negative_prompt = st.text_input(
                "Negative prompt (optional):",
                value=default_negative if default_negative else "blurry, low quality, distorted, deformed, ugly"
            )
            
            # Stable Diffusion settings
            col_sd1, col_sd2 = st.columns(2)
            with col_sd1:
                sd_steps = st.slider("Diffusion Steps", 4, 25, 8, 1,
                                    help="Number of denoising steps (higher = better quality, slower)",
                                    key="replace_sd_steps")
            with col_sd2:
                mask_dilation = st.slider("Mask Dilation", 1, 30, 12, 1,
                                         help="Expand mask for better edge blending",
                                         key="replace_dilation")
            
            if st.button("🔄 Replace Object", type="primary", disabled=(st.session_state.current_mask is None)):
                if replace_prompt:
                    with st.spinner("Replacing object with Stable Diffusion..."):
                        result = perform_replacement(
                            image_np, st.session_state.current_mask,
                            replace_prompt, negative_prompt, get_device(),
                            steps=sd_steps, dilation=mask_dilation
                        )
                        st.session_state.result_image = result
                else:
                    st.warning("Please enter a replacement prompt.")
        
        elif task == "background":
            st.markdown("**Keep the selected object, replace the background.**")
            
            # Pre-fill with agent-extracted prompt if available
            default_bg = st.session_state.get('agent_diffusion_prompt', '')
            
            if default_bg and is_agent_available():
                st.success(f"🤖 AI suggested: *{default_bg}*")
            
            bg_prompt = st.text_input(
                "Describe the new background:",
                value=default_bg if default_bg else "",
                placeholder="e.g., 'a beautiful sunset beach', 'modern office interior'"
            )
            
            # Stable Diffusion settings
            col_sd1, col_sd2 = st.columns(2)
            with col_sd1:
                bg_sd_steps = st.slider("Diffusion Steps", 4, 25, 20, 1,
                                       help="Number of denoising steps (higher = better quality, slower)",
                                       key="bg_sd_steps")
            with col_sd2:
                bg_dilation = st.slider("Mask Dilation", 1, 30, 3, 1,
                                       help="Expand mask for better edge blending",
                                       key="bg_dilation")
            
            if st.button("🌅 Change Background", type="primary", disabled=(st.session_state.current_mask is None)):
                if bg_prompt:
                    with st.spinner("Generating new background with Stable Diffusion..."):
                        result = perform_background_fill(
                            image_np, st.session_state.current_mask,
                            bg_prompt, get_device(),
                            steps=bg_sd_steps, dilation=bg_dilation
                        )
                        st.session_state.result_image = result
                else:
                    st.warning("Please enter a background prompt.")
        
        # Show result
        if st.session_state.result_image is not None:
            st.markdown("---")
            st.subheader("Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(st.session_state.result_image, caption="Result", use_container_width=True)
            
            # Download button
            result_pil = image_to_pil(st.session_state.result_image)
            buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            result_pil.save(buf.name)
            with open(buf.name, 'rb') as f:
                st.download_button(
                    "📥 Download Result",
                    f.read(),
                    file_name="edited_image.png",
                    mime="image/png"
                )


def render_relighting_page():
    """Render the neural relighting feature page."""
    st.header("💡 Neural Relighting")
    
    # Cloud demo notice
    if IS_STREAMLIT_CLOUD:
        st.info("""
        ☁️ **Cloud Demo Mode** - This is a UI demonstration. For full functionality with AI models:
        1. Clone the repo: `git clone https://github.com/team24-stack/adobe.git`
        2. Download model weights from Google Drive (see README)
        3. Run locally: `streamlit run app.py`
        
        📹 [Watch Demo Video](https://github.com/team24-stack/adobe) | 📄 [View Documentation](https://github.com/team24-stack/adobe/blob/main/workflows/README.md)
        """)
    
    st.markdown("""
    <div class="info-box">
    <b>Capabilities:</b> Change the lighting direction and intensity of any image 
    using physics-based rendering and neural network refinement.
    </div>
    """, unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key="relight_upload"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        img_h, img_w = image_np.shape[:2]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Initialize session state for light direction from arrow
        if "relight_arrow_start" not in st.session_state:
            st.session_state.relight_arrow_start = None
        if "relight_arrow_end" not in st.session_state:
            st.session_state.relight_arrow_end = None
        if "relight_light_x" not in st.session_state:
            st.session_state.relight_light_x = 0.0
        if "relight_light_y" not in st.session_state:
            st.session_state.relight_light_y = 0.0
        
        # Light direction controls
        st.subheader("🎯 Light Direction")
        
        # Method selection
        direction_method = st.radio(
            "Set light direction by:",
            ["🎚️ Sliders", "➡️ Draw Arrow"],
            horizontal=True,
            key="light_direction_method"
        )
        
        if "Draw Arrow" in direction_method:
            st.markdown("**Draw an arrow** to set light direction (X, Y). Light comes FROM the arrow direction.")
            
            if CANVAS_AVAILABLE:
                # Calculate canvas size
                canvas_width = min(500, img_w)
                scale = canvas_width / img_w
                canvas_height = int(img_h * scale)
                
                canvas_image = cv2.resize(image_np, (canvas_width, canvas_height))
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 0, 0.3)",
                    stroke_width=3,
                    stroke_color="#FFFF00",
                    background_image=Image.fromarray(canvas_image),
                    update_streamlit=False,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="line",
                    key="arrow_canvas",
                )
                
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    if objects:
                        # Get the last drawn line (arrow)
                        line = objects[-1]
                        if line["type"] == "line":
                            # Extract start and end points
                            x1 = line.get("x1", 0) + line.get("left", 0)
                            y1 = line.get("y1", 0) + line.get("top", 0)
                            x2 = line.get("x2", 0) + line.get("left", 0)
                            y2 = line.get("y2", 0) + line.get("top", 0)
                            
                            # Scale back to image coordinates
                            start_x, start_y = x1 / scale, y1 / scale
                            end_x, end_y = x2 / scale, y2 / scale
                            
                            # Calculate direction (light comes FROM arrow direction)
                            dx = end_x - start_x
                            dy = end_y - start_y
                            
                            # Normalize to -1 to 1 range
                            max_dim = max(img_w, img_h)
                            light_x = np.clip(-dx / max_dim * 2, -1.0, 1.0)
                            light_y = np.clip(-dy / max_dim * 2, -1.0, 1.0)
                            
                            st.session_state.relight_light_x = float(light_x)
                            st.session_state.relight_light_y = float(light_y)
                            
                            st.success(f"Arrow direction: X={light_x:.2f}, Y={light_y:.2f}")
            else:
                st.warning("Canvas not available. Use manual coordinate input:")
                col_arrow1, col_arrow2 = st.columns(2)
                with col_arrow1:
                    start_x = st.number_input("Start X", 0, img_w, img_w//2, key="arrow_start_x")
                    start_y = st.number_input("Start Y", 0, img_h, img_h//2, key="arrow_start_y")
                with col_arrow2:
                    end_x = st.number_input("End X", 0, img_w, img_w//2 + 100, key="arrow_end_x")
                    end_y = st.number_input("End Y", 0, img_h, img_h//2, key="arrow_end_y")
                
                if st.button("Calculate Direction", key="calc_arrow_dir"):
                    dx = end_x - start_x
                    dy = end_y - start_y
                    max_dim = max(img_w, img_h)
                    st.session_state.relight_light_x = float(np.clip(-dx / max_dim * 2, -1.0, 1.0))
                    st.session_state.relight_light_y = float(np.clip(-dy / max_dim * 2, -1.0, 1.0))
            
            # Use arrow-derived values
            light_x = st.session_state.relight_light_x
            light_y = st.session_state.relight_light_y
            
            st.caption(f"Current X: {light_x:.2f}, Y: {light_y:.2f}")
        else:
            # Slider method
            col_x, col_y = st.columns(2)
            with col_x:
                light_x = st.slider("X (Left ↔ Right)", -1.0, 1.0, 0.0, 0.05,
                                   help="Negative = light from left, Positive = light from right",
                                   key="light_x_slider")
            with col_y:
                light_y = st.slider("Y (Top ↔ Bottom)", -1.0, 1.0, -0.5, 0.05,
                                   help="Negative = light from top, Positive = light from bottom",
                                   key="light_y_slider")
        
        # Z (Depth) slider - always shown
        st.markdown("---")
        light_z = st.slider("🔹 Depth (Z)", -1.0, 1.0, -0.5, 0.05,
                           help="Z depth of light. Negative = in front, Positive = behind",
                           key="light_z_slider")
        
        # Intensity control
        intensity = st.slider("💡 Intensity", 0.0, 3.0, 1.0, 0.1,
                             help="Overall brightness of the lighting",
                             key="light_intensity_slider")
        
        # Light presets
        st.subheader("⚡ Light Presets")
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        preset_light_x, preset_light_y, preset_light_z = light_x, light_y, light_z
        
        with preset_col1:
            if st.button("☀️ Top", key="preset_top"):
                preset_light_x, preset_light_y, preset_light_z = 0.0, -1.0, -0.5
        with preset_col2:
            if st.button("🌅 Left", key="preset_left"):
                preset_light_x, preset_light_y, preset_light_z = -1.0, 0.0, -0.5
        with preset_col3:
            if st.button("🌄 Right", key="preset_right"):
                preset_light_x, preset_light_y, preset_light_z = 1.0, 0.0, -0.5
        with preset_col4:
            if st.button("🔦 Front", key="preset_front"):
                preset_light_x, preset_light_y, preset_light_z = 0.0, 0.0, -1.0
        
        # Use preset values if clicked (check if different from current)
        if (preset_light_x, preset_light_y, preset_light_z) != (light_x, light_y, light_z):
            light_x, light_y, light_z = preset_light_x, preset_light_y, preset_light_z
        
        # Show current light vector
        light_vec = [light_x, light_y, light_z]
        norm = np.linalg.norm(light_vec) + 1e-9
        normalized = np.array(light_vec) / norm
        st.caption(f"📍 Light Vector: [{normalized[0]:.2f}, {normalized[1]:.2f}, {normalized[2]:.2f}] | Intensity: {intensity:.1f}")
        
        # Apply relighting
        if st.button("✨ Apply Relighting", type="primary", key="apply_relight_btn"):
            with st.spinner("Computing depth, normals, and relighting..."):
                # Load models
                model = load_relight_model()
                midas, midas_transforms = load_midas()
                
                if model and midas:
                    result, physics, depth, normals = compute_relighting(
                        image_bgr, model, midas, midas_transforms,
                        light_vec, intensity, get_device()
                    )
                    
                    # Store results
                    st.session_state.relight_result = result
                    st.session_state.physics_result = (physics * 255).astype(np.uint8)
                    st.session_state.depth_result = depth
                    st.session_state.normals_result = normals
        
        # Show results
        if 'relight_result' in st.session_state and st.session_state.relight_result is not None:
            with col2:
                st.subheader("Relit Result")
                st.image(st.session_state.relight_result, use_container_width=True)
            
            # Show intermediate results
            st.markdown("---")
            st.subheader("Intermediate Results")
            
            tab1, tab2, tab3 = st.tabs(["🎨 Physics Render", "🗺️ Depth Map", "📐 Surface Normals"])
            
            with tab1:
                if 'physics_result' in st.session_state:
                    st.image(st.session_state.physics_result, caption="Physics-based rendering (Lambertian + Blinn-Phong)")
            
            with tab2:
                if 'depth_result' in st.session_state:
                    depth_vis = (st.session_state.depth_result * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
                    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
                    st.image(depth_colored, caption="MiDaS Depth Estimation")
            
            with tab3:
                if 'normals_result' in st.session_state:
                    normals_vis = ((st.session_state.normals_result + 1) / 2 * 255).astype(np.uint8)
                    st.image(normals_vis, caption="Surface Normals (RGB encoded)")
            
            # Download button
            result_pil = image_to_pil(st.session_state.relight_result)
            buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            result_pil.save(buf.name)
            with open(buf.name, 'rb') as f:
                st.download_button(
                    "📥 Download Relit Image",
                    f.read(),
                    file_name="relit_image.png",
                    mime="image/png"
                )


if __name__ == "__main__":
    main()
