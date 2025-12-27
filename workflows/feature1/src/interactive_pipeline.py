"""
Interactive Image Editing Pipeline

Provides an interactive GUI for:
- Object Removal (using LAMA)
- Object Replacement (using Stable Diffusion with LCM)
- Background Fill (using Stable Diffusion with LCM)

Supports two modes:
- Manual: Box, Click, or Sketch-based mask selection
- Agent: Text prompt with CLIP-based automatic mask generation

Usage:
    python interactive_pipeline.py --image ./images/test.jpg
"""

import os
import sys
import cv2
import numpy as np
import torch
import logging
from enum import Enum
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

# Disable HF transfer to avoid errors
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Suppress verbose logging from LAMA and other modules
logging.getLogger("saicinpainting").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

# Default paths (relative to workspace root)
DEFAULT_SAM_CKPT = "./feature1/checkpoints/mobile_sam.pt"
DEFAULT_LAMA_CONFIG = "./feature1/lama/configs/prediction/default.yaml"
DEFAULT_LAMA_CKPT = "./feature1/checkpoints/lama-dilated"


# =============================================================================
# ENUMS AND CONFIG
# =============================================================================

class Task(Enum):
    REMOVE = "remove"
    REPLACE = "replace"
    BACKGROUND = "background"


class Mode(Enum):
    MANUAL = "manual"
    AGENT = "agent"


class ManualPromptType(Enum):
    BOX = "box"
    CLICK = "click"
    SKETCH = "sketch"


# Global configuration
DILATION_KERNEL_SIZE = 12      # Dilation size for remove/replace operations
DILATION_BACKGROUND = 3        # Dilation size for background fill (smaller to preserve edges)
SD_INFERENCE_STEPS = 4         # Number of inference steps for SD with LCM
SD_GUIDANCE_SCALE = 4.0        # Guidance scale for LCM


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MaskState:
    """State management for mask editing with undo support."""
    mask: np.ndarray
    history: List[np.ndarray] = field(default_factory=list)
    max_history: int = 20
    
    def save_state(self):
        """Save current mask state for undo."""
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(self.mask.copy())
    
    def undo(self) -> bool:
        """Undo last mask change. Returns True if successful."""
        if self.history:
            self.mask = self.history.pop()
            return True
        return False
    
    def reset(self, shape: Tuple[int, int]):
        """Reset mask to empty."""
        self.mask = np.zeros(shape, dtype=np.uint8)
        self.history.clear()
    
    def add_mask(self, new_mask: np.ndarray):
        """Add (union) new mask to current mask."""
        self.save_state()
        self.mask = np.maximum(self.mask, new_mask)
    
    def subtract_mask(self, erase_mask: np.ndarray):
        """Subtract (erase) from current mask."""
        self.save_state()
        self.mask[erase_mask > 0] = 0


# =============================================================================
# INTERACTIVE PIPELINE CLASS
# =============================================================================

class InteractivePipeline:
    """
    Interactive image editing pipeline supporting:
    - Object removal with LAMA
    - Object replacement with SD + LCM
    - Background fill with SD + LCM
    """
    
    def __init__(
        self,
        sam_checkpoint: str = DEFAULT_SAM_CKPT,
        lama_config: str = DEFAULT_LAMA_CONFIG,
        lama_checkpoint: str = DEFAULT_LAMA_CKPT,
        device: str = "cpu"
    ):
        self.sam_checkpoint = sam_checkpoint
        self.lama_config = lama_config
        self.lama_checkpoint = lama_checkpoint
        self.device = device
        
        # Lazy loaded models
        self._sam_predictor = None
        self._lama_model = None
        self._sd_pipeline = None
        self._agent_chain = None
        
        # State
        self.image = None
        self.original_image = None
        self.mask_state = None
        self.window_name = "Interactive Pipeline"
        
        # Manual mode state
        self.drawing = False
        self.current_prompt_type = ManualPromptType.BOX
        self.brush_size = 20
        self.click_point = None  # For single click mode
        self.box_start = None
        self.box_end = None
        self.sketch_points = []  # For sketch mode - collect points for SAM
        
    # =========================================================================
    # MODEL LOADING (Lazy)
    # =========================================================================
    
    @property
    def sam_predictor(self):
        """Lazy load SAM predictor."""
        if self._sam_predictor is None:
            print("\n[Loading SAM model...]")
            from .mobilesamsegment import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_t"](checkpoint=self.sam_checkpoint)
            sam.to(self.device)
            sam.eval()
            self._sam_predictor = SamPredictor(sam)
            print("[SAM model loaded]")
        return self._sam_predictor
    
    @property
    def lama_model(self):
        """Lazy load LAMA model."""
        if self._lama_model is None:
            print("\n[Loading LAMA model...]")
            # Suppress verbose logging during LAMA loading
            import logging
            logging.getLogger().setLevel(logging.WARNING)
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            
            from .inpaint_by_lama import build_lama_model
            self._lama_model = build_lama_model(
                self.lama_config, 
                self.lama_checkpoint, 
                device=self.device
            )
            print("[LAMA model loaded]")
        return self._lama_model
    
    @property
    def sd_pipeline(self):
        """Lazy load Stable Diffusion pipeline with LCM."""
        if self._sd_pipeline is None:
            print("\n[Loading Stable Diffusion with LCM...]")
            from .inpaint_by_sd import build_sd_inpaint_model
            self._sd_pipeline = build_sd_inpaint_model(
                device=self.device,
                use_lcm=True,
                download_to_checkpoints=True
            )
            print("[SD + LCM loaded]")
        return self._sd_pipeline
    
    @property
    def agent_chain(self):
        """Lazy load agent chain for prompt processing."""
        if self._agent_chain is None:
            try:
                from .agent_decision import create_agent_chain, check_api_key
                check_api_key()
                self._agent_chain = create_agent_chain()
                print("[Agent loaded]")
            except Exception as e:
                print(f"[Agent not available: {e}]")
                self._agent_chain = False  # Mark as unavailable
        return self._agent_chain
    
    def preload_models(self, tasks: List[str] = None):
        """
        Preload models at startup to avoid delays during editing.
        
        Args:
            tasks: List of tasks to preload models for. 
                   Options: ['sam', 'lama', 'sd', 'agent']
                   If None, loads SAM and LAMA (most common).
        """
        if tasks is None:
            tasks = ['sam', 'lama']
        
        print("\n[Preloading models...]")
        
        if 'sam' in tasks:
            _ = self.sam_predictor
        
        if 'lama' in tasks:
            _ = self.lama_model
        
        if 'sd' in tasks:
            _ = self.sd_pipeline
        
        if 'agent' in tasks:
            _ = self.agent_chain
        
        print("[All models preloaded]\n")
    
    # =========================================================================
    # AGENT FUNCTIONS
    # =========================================================================
    
    def process_full_prompt_with_agent(self, user_prompt: str) -> dict:
        """
        Process full user prompt with agent to extract:
        - task: remove, replace, background_fill
        - clip_prompt: Object description for CLIP+SAM segmentation
        - diffusion_prompt: Positive prompt for SD (None for remove)
        - negative_diffusion_prompt: Negative prompt for SD (None for remove)
        
        Returns dict with all fields, or fallback values if agent unavailable.
        """
        if self.agent_chain and self.agent_chain is not False:
            try:
                from agent_decision import process_user_prompt
                result = process_user_prompt(self.agent_chain, user_prompt)
                if "error" not in result:
                    return result
                print(f"  Agent error: {result.get('error')}")
            except Exception as e:
                print(f"  Agent processing failed: {e}")
        
        # Fallback: parse user input manually when agent unavailable
        return self._fallback_parse_prompt(user_prompt)
    
    def _fallback_parse_prompt(self, user_prompt: str) -> dict:
        """
        Fallback parsing when agent is unavailable.
        Extracts task and clip_prompt from user input using simple rules.
        """
        prompt_lower = user_prompt.lower().strip()
        
        # Detect task type
        task = "replace"  # default
        if any(kw in prompt_lower for kw in ["remove", "delete", "erase", "get rid of", "take out", "clear"]):
            task = "remove"
        elif any(kw in prompt_lower for kw in ["background", "scene", "put in", "place in"]):
            task = "background_fill"
        elif any(kw in prompt_lower for kw in ["replace", "change", "turn into", "swap", "convert"]):
            task = "replace"
        
        # Extract clip_prompt by removing action keywords
        clip_prompt = user_prompt
        action_phrases = [
            "remove the ", "remove ", "delete the ", "delete ", 
            "erase the ", "erase ", "get rid of the ", "get rid of ",
            "take out the ", "take out ", "clear the ", "clear ",
            "replace the ", "replace ", "change the ", "change ",
            "swap the ", "swap ", "convert the ", "convert ",
            "turn the ", "turn ", "make the ", "make ",
            "change the background to ", "change background to ",
            "put the ", "put ", "place the ", "place ",
            "background to ", "scene to "
        ]
        
        for phrase in action_phrases:
            if prompt_lower.startswith(phrase):
                clip_prompt = user_prompt[len(phrase):]
                break
        
        # For replace tasks, try to split at "with", "to", "into"
        diffusion_prompt = None
        negative_prompt = None
        
        if task == "replace":
            for separator in [" with ", " to ", " into "]:
                if separator in clip_prompt.lower():
                    parts = clip_prompt.lower().split(separator, 1)
                    # First part is the object to find
                    clip_prompt = parts[0].strip()
                    # Second part is what to replace with
                    diffusion_prompt = parts[1].strip()
                    negative_prompt = "blurry, low quality, distorted, deformed, ugly"
                    break
        elif task == "background_fill":
            # For background, clip_prompt should be the subject to keep
            # and diffusion_prompt is the new background
            diffusion_prompt = clip_prompt
            clip_prompt = "main subject"  # Will be refined by user or use default
            negative_prompt = "blurry, low quality, distorted, deformed, ugly"
        
        # For remove task: no diffusion prompts
        if task == "remove":
            diffusion_prompt = None
            negative_prompt = None
        
        print(f"  [Fallback parsing] Task: {task}, CLIP: '{clip_prompt}'")
        
        return {
            "task": task,
            "clip_prompt": clip_prompt.strip(),
            "diffusion_prompt": diffusion_prompt,
            "negative_diffusion_prompt": negative_prompt
        }
    
    def extract_diffusion_prompts_with_agent(self, user_prompt: str) -> dict:
        """
        Extract only diffusion prompts from user prompt (for manual mode).
        
        Returns:
            dict with diffusion_prompt and negative_diffusion_prompt
        """
        try:
            from agent_decision import extract_diffusion_prompts
            result = extract_diffusion_prompts(user_prompt)
            print(f"  Diffusion prompt: {result.get('diffusion_prompt', '')[:60]}...")
            print(f"  Negative prompt: {result.get('negative_diffusion_prompt', '')[:60]}...")
            return result
        except Exception as e:
            print(f"  Agent extraction failed: {e}")
            # Fallback
            return {
                "diffusion_prompt": user_prompt,
                "negative_diffusion_prompt": "blurry, low quality, distorted, deformed, ugly"
            }
    
    def get_diffusion_prompt_from_agent(self, user_prompt: str, task_type: str) -> str:
        """
        Get enhanced diffusion prompt from agent.
        Falls back to direct input if agent unavailable.
        """
        try:
            from agent_decision import reframe_prompt_for_diffusion
            enhanced = reframe_prompt_for_diffusion(user_prompt, task_type)
            print(f"  Enhanced prompt: {enhanced}")
            return enhanced
        except Exception as e:
            print(f"  Agent unavailable ({e}), using direct prompt")
            return user_prompt
    
    # =========================================================================
    # MASK GENERATION
    # =========================================================================
    
    def generate_mask_from_box(self, box: np.ndarray) -> np.ndarray:
        """Generate mask from bounding box using SAM."""
        self.sam_predictor.set_image(self.image)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False
        )
        return (masks[0] * 255).astype(np.uint8)
    
    def generate_mask_from_points(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Generate mask from points using SAM."""
        self.sam_predictor.set_image(self.image)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            multimask_output=True
        )
        # Return best mask
        best_idx = np.argmax(scores)
        return (masks[best_idx] * 255).astype(np.uint8)
    
    def generate_mask_from_text(self, text_prompt: str) -> Optional[np.ndarray]:
        """Generate mask from text prompt using CLIP + SAM."""
        try:
            from .clip_masking import find_object_with_text, CLIP_AVAILABLE
            if not CLIP_AVAILABLE:
                print("  CLIP not available for text-based masking")
                return None
            
            print(f"  Searching for: '{text_prompt}'")
            mask = find_object_with_text(
                self.image,
                text_prompt,
                self.sam_checkpoint,
                model_type="vit_t",
                device=self.device
            )
            return mask
        except Exception as e:
            print(f"  Text-based mask generation failed: {e}")
            return None
    
    # =========================================================================
    # MASK PROCESSING
    # =========================================================================
    
    def dilate_mask(self, mask: np.ndarray, kernel_size: int = DILATION_KERNEL_SIZE) -> np.ndarray:
        """Dilate mask with specified kernel size."""
        if kernel_size <= 0:
            return mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)
    
    # =========================================================================
    # INPAINTING OPERATIONS
    # =========================================================================
    
    def perform_removal(self, mask: np.ndarray) -> np.ndarray:
        """Remove object using LAMA inpainting."""
        print("\n[Performing object removal with LAMA...]")
        
        # Debug: show mask coverage
        total_pixels = mask.shape[0] * mask.shape[1]
        masked_pixels = np.sum(mask > 0)
        print(f"  Mask coverage: {masked_pixels}/{total_pixels} pixels ({100*masked_pixels/total_pixels:.1f}%)")
        
        # Dilate mask for better edge coverage
        dilated_mask = self.dilate_mask(mask, DILATION_KERNEL_SIZE)
        dilated_pixels = np.sum(dilated_mask > 0)
        print(f"  After dilation ({DILATION_KERNEL_SIZE}px): {dilated_pixels}/{total_pixels} pixels ({100*dilated_pixels/total_pixels:.1f}%)")
        
        from .inpaint_by_lama import inpaint_img_with_builded_lama
        result = inpaint_img_with_builded_lama(
            self.lama_model,
            self.image,
            dilated_mask,
            device=self.device
        )
        
        print("[Removal complete]")
        return result
    
    def perform_replacement(self, mask: np.ndarray, prompt: str, negative_prompt: str = None) -> np.ndarray:
        """Replace object using Stable Diffusion with LCM."""
        print(f"\n[Performing object replacement with SD + LCM...]")
        print(f"  Prompt: {prompt}")
        if negative_prompt:
            print(f"  Negative: {negative_prompt[:60]}...")
        print(f"  Steps: {SD_INFERENCE_STEPS}")
        
        # Debug: show mask coverage
        total_pixels = mask.shape[0] * mask.shape[1]
        masked_pixels = np.sum(mask > 0)
        print(f"  Mask coverage: {masked_pixels}/{total_pixels} pixels ({100*masked_pixels/total_pixels:.1f}%)")
        
        # Dilate mask for better edge coverage
        dilated_mask = self.dilate_mask(mask, DILATION_KERNEL_SIZE)
        dilated_pixels = np.sum(dilated_mask > 0)
        print(f"  After dilation ({DILATION_KERNEL_SIZE}px): {dilated_pixels}/{total_pixels} pixels ({100*dilated_pixels/total_pixels:.1f}%)")
        
        # Use dedicated replace function with proper resize/pad/composite
        from .inpaint_by_sd import replace_img_with_sd
        result = replace_img_with_sd(
            self.image,
            dilated_mask,
            prompt,
            negative_prompt=negative_prompt or "blurry, low quality, distorted, deformed, ugly",
            step=SD_INFERENCE_STEPS,
            device=self.device
        )
        
        print("[Replacement complete]")
        return result
    
    def perform_background_fill(self, mask: np.ndarray, prompt: str, negative_prompt: str = None) -> np.ndarray:
        """Fill background using Stable Diffusion with LCM, keeping foreground object."""
        print(f"\n[Performing background fill with SD + LCM...]")
        print(f"  Prompt: {prompt}")
        if negative_prompt:
            print(f"  Negative: {negative_prompt[:60]}...")
        print(f"  Steps: {SD_INFERENCE_STEPS}")
        
        # Debug: show mask coverage
        total_pixels = mask.shape[0] * mask.shape[1]
        foreground_pixels = np.sum(mask > 0)
        print(f"  Foreground mask (object to keep): {foreground_pixels}/{total_pixels} pixels ({100*foreground_pixels/total_pixels:.1f}%)")
        
        background_pixels = total_pixels - foreground_pixels
        print(f"  Background to fill: {background_pixels}/{total_pixels} pixels ({100*background_pixels/total_pixels:.1f}%)")
        
        # Use dedicated background fill function that does proper compositing
        from .inpaint_by_sd import fill_background_with_sd
        result = fill_background_with_sd(
            self.image,
            mask,  # WHITE = object to KEEP
            prompt,
            step=20,  # More steps for better background quality
            device=self.device
        )
        
        print("[Background fill complete]")
        return result
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def get_display_image(self, show_mask: bool = True) -> np.ndarray:
        """Get image with mask overlay for display."""
        display = self.image.copy()
        
        if show_mask and self.mask_state and np.any(self.mask_state.mask > 0):
            # Create red overlay for mask
            overlay = display.copy()
            overlay[self.mask_state.mask > 0] = [0, 0, 255]  # Red in BGR
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        return display
    
    def show_controls(self, task: Task, mode: Mode, prompt_type: ManualPromptType = None):
        """Print current controls to console."""
        print("\n" + "=" * 50)
        print(f"  Task: {task.value.upper()} | Mode: {mode.value.upper()}")
        if mode == Mode.MANUAL and prompt_type:
            print(f"  Selection: {prompt_type.value.upper()}")
        print("=" * 50)
        
        # Task-specific instructions
        if task == Task.REMOVE:
            print("  Select the OBJECT to REMOVE")
        elif task == Task.REPLACE:
            print("  Select the OBJECT to REPLACE")
        elif task == Task.BACKGROUND:
            print("  Select the FOREGROUND OBJECT to KEEP")
            print("  (Background will be replaced)")
        
        print("-" * 50)
        print("  [z] Undo mask    [r] Reset mask")
        print("  [a] Apply & Inpaint")
        print("  [q] Quit")
        if mode == Mode.MANUAL:
            print("  [b] Box mode  [c] Click mode  [s] Sketch mode")
        print("=" * 50)
    
    # =========================================================================
    # MOUSE CALLBACKS
    # =========================================================================
    
    def mouse_callback_box(self, event, x, y, flags, param):
        """Mouse callback for box selection mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.box_start = (x, y)
            self.box_end = None
            self.drawing = True
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.box_end = (x, y)
            # Show preview
            display = self.get_display_image()
            cv2.rectangle(display, self.box_start, self.box_end, (0, 255, 0), 2)
            cv2.imshow(self.window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.box_end = (x, y)
            
            if self.box_start and self.box_end:
                x1, y1 = self.box_start
                x2, y2 = self.box_end
                box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                
                if box[2] - box[0] > 5 and box[3] - box[1] > 5:
                    print("  Generating mask from box...")
                    mask = self.generate_mask_from_box(box)
                    self.mask_state.add_mask(mask)
                    print("  Mask added. Press [a] to apply or continue editing.")
            
            self.box_start = None
            self.box_end = None
    
    def mouse_callback_click(self, event, x, y, flags, param):
        """Mouse callback for single click selection mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            print(f"  Generating mask from click at ({x}, {y})...")
            # Generate mask from single foreground point
            points = np.array([[x, y]])
            labels = np.array([1])  # Foreground
            mask = self.generate_mask_from_points(points, labels)
            self.mask_state.add_mask(mask)
            print("  Mask added. Press [a] to apply or click another object.")
    
    def mouse_callback_sketch(self, event, x, y, flags, param):
        """Mouse callback for sketch mode - collects stroke points for SAM."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.sketch_points.append((x, y))
            # Draw visual feedback
            self._draw_sketch_point(x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Sample points along stroke (every few pixels)
            if self.sketch_points:
                last_x, last_y = self.sketch_points[-1]
                dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                if dist > 10:  # Sample every 10 pixels
                    self.sketch_points.append((x, y))
                    self._draw_sketch_point(x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Generate mask from bounding box (start and end points) using SAM
            if len(self.sketch_points) >= 2:
                self._generate_mask_from_sketch_box()
    
    def _draw_sketch_point(self, x: int, y: int):
        """Draw visual feedback for sketch point."""
        # Draw on a temporary overlay for visual feedback
        cv2.circle(self.mask_state.mask, (x, y), 3, 128, -1)  # Gray dots for preview
    
    def _generate_mask_from_sketch_box(self):
        """Generate SAM mask from bounding box extracted from all sketch points."""
        if len(self.sketch_points) < 2:
            return
        
        # Extract bounding box from all sketch points (topmost, bottommost, leftmost, rightmost)
        points_np = np.array(self.sketch_points)
        x1 = int(np.min(points_np[:, 0]))
        y1 = int(np.min(points_np[:, 1]))
        x2 = int(np.max(points_np[:, 0]))
        y2 = int(np.max(points_np[:, 1]))
        
        # Add small padding
        padding = 5
        h, w = self.image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w - 1, x2 + padding)
        y2 = min(h - 1, y2 + padding)
        
        # Ensure box has minimum size
        if x2 - x1 < 10:
            x2 = x1 + 10
        if y2 - y1 < 10:
            y2 = y1 + 10
        
        print(f"  Generating mask from sketch bounding box: [{x1}, {y1}, {x2}, {y2}]")
        
        # Clear preview dots
        self.mask_state.mask[self.mask_state.mask == 128] = 0
        
        input_box = np.array([x1, y1, x2, y2])
        
        # Generate mask using SAM with bounding box
        try:
            self.sam_predictor.set_image(self.image)
            masks, scores, _ = self.sam_predictor.predict(
                box=input_box,
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            new_mask = (masks[best_idx] * 255).astype(np.uint8)
            self.mask_state.add_mask(new_mask)
            print(f"  Mask generated! Score: {scores[best_idx]:.3f}")
        except Exception as e:
            print(f"  Error generating mask: {e}")
        
        # Clear sketch points for next sketch
        self.sketch_points = []
    
    def _paint_at(self, x: int, y: int):
        """Paint at position (legacy - kept for direct paint mode if needed)."""
        cv2.circle(self.mask_state.mask, (x, y), self.brush_size, 255, -1)
    
    # =========================================================================
    # MANUAL MODE
    # =========================================================================
    
    def run_manual_mode(self, task: Task) -> Optional[Tuple[np.ndarray, Optional[str], Optional[str]]]:
        """
        Run manual mask selection mode.
        
        Returns:
            Tuple of (mask, diffusion_prompt, negative_diffusion_prompt) or None if cancelled.
            diffusion_prompt and negative_diffusion_prompt are None for REMOVE task.
        """
        h, w = self.image.shape[:2]
        self.mask_state = MaskState(mask=np.zeros((h, w), dtype=np.uint8))
        self.current_prompt_type = ManualPromptType.BOX
        self.click_point = None
        diffusion_prompt = None
        negative_diffusion_prompt = None
        
        cv2.namedWindow(self.window_name)
        self.show_controls(task, Mode.MANUAL, self.current_prompt_type)
        
        while True:
            # Set appropriate mouse callback
            if self.current_prompt_type == ManualPromptType.BOX:
                cv2.setMouseCallback(self.window_name, self.mouse_callback_box)
            elif self.current_prompt_type == ManualPromptType.CLICK:
                cv2.setMouseCallback(self.window_name, self.mouse_callback_click)
            elif self.current_prompt_type == ManualPromptType.SKETCH:
                cv2.setMouseCallback(self.window_name, self.mouse_callback_sketch)
            
            # Display
            display = self.get_display_image()
            
            # Show last click point for click mode
            if self.current_prompt_type == ManualPromptType.CLICK and self.click_point:
                cv2.circle(display, self.click_point, 5, (0, 255, 0), -1)
            
            # Show brush info for sketch mode
            if self.current_prompt_type == ManualPromptType.SKETCH:
                cv2.putText(display, f"Brush: {self.brush_size}px", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                print("  Cancelled")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('z'):  # Undo
                if self.mask_state.undo():
                    print("  Undo successful")
                else:
                    print("  Nothing to undo")
            
            elif key == ord('r'):  # Reset
                self.mask_state.reset((h, w))
                self.click_point = None
                print("  Mask reset")
            
            elif key == ord('a'):  # Apply
                if np.any(self.mask_state.mask > 0):
                    # For Replace/Background: ask for prompt after mask is selected
                    if task in [Task.REPLACE, Task.BACKGROUND]:
                        cv2.destroyAllWindows()
                        print("\n" + "-" * 50)
                        if task == Task.BACKGROUND:
                            print("  You selected the FOREGROUND object to keep.")
                            print("  Now enter description for the NEW BACKGROUND:")
                        else:
                            print("  Mask selected! Now enter description for the new content:")
                        user_input = input("  > ").strip()
                        if not user_input:
                            print("  No prompt provided, using default")
                            user_input = "high quality, detailed, realistic"
                        
                        # Use agent to extract diffusion prompts
                        print("\n  Extracting prompts with agent...")
                        prompts = self.extract_diffusion_prompts_with_agent(user_input)
                        diffusion_prompt = prompts.get("diffusion_prompt", user_input)
                        negative_diffusion_prompt = prompts.get("negative_diffusion_prompt")
                    else:
                        cv2.destroyAllWindows()
                    
                    print("  Applying mask...")
                    return (self.mask_state.mask, diffusion_prompt, negative_diffusion_prompt)
                else:
                    print("  No mask to apply. Draw a mask first.")
            
            elif key == ord('b'):  # Box mode
                self.current_prompt_type = ManualPromptType.BOX
                print("  Switched to BOX mode")
            
            elif key == ord('c'):  # Click mode
                self.current_prompt_type = ManualPromptType.CLICK
                self.click_point = None
                print("  Switched to CLICK mode")
            
            elif key == ord('s'):  # Sketch mode
                self.current_prompt_type = ManualPromptType.SKETCH
                self.sketch_points = []  # Clear any previous sketch points
                print("  Switched to SKETCH mode (draw stroke → SAM generates mask)")
        
        cv2.destroyAllWindows()
        return None
    
    # =========================================================================
    # AGENT MODE
    # =========================================================================
    
    def run_agent_mode(self, task: Task) -> Optional[Tuple[np.ndarray, Optional[str], Optional[str]]]:
        """
        Run agent-assisted mask selection mode.
        
        Workflow:
        1. User enters text prompt
        2. Agent extracts: task, clip_prompt, diffusion_prompt, negative_diffusion_prompt
        3. CLIP + SAM uses clip_prompt for mask generation
        4. For remove: diffusion prompts are null (LAMA only)
        5. For replace/background: diffusion prompts passed to SD
        
        Returns:
            Tuple of (mask, diffusion_prompt, negative_diffusion_prompt) or None if cancelled.
        """
        h, w = self.image.shape[:2]
        self.mask_state = MaskState(mask=np.zeros((h, w), dtype=np.uint8))
        
        print("\n" + "-" * 50)
        print("  Enter your request (e.g., 'remove the car on the left'):")
        print("  Or for replace: 'replace the dog with a fluffy cat'")
        print("  Or for background: 'change the background to a beach sunset'")
        user_input = input("  > ").strip()
        
        if not user_input:
            print("  No input provided")
            return None
        
        # Process with agent - extract all fields
        print("\n  Processing with agent...")
        agent_result = self.process_full_prompt_with_agent(user_input)
        
        # Extract fields
        detected_task = agent_result.get("task", task.value)
        clip_prompt = agent_result.get("clip_prompt", user_input)
        diffusion_prompt = agent_result.get("diffusion_prompt")
        negative_diffusion_prompt = agent_result.get("negative_diffusion_prompt")
        
        print(f"\n  Agent extracted:")
        print(f"    Task: {detected_task}")
        print(f"    CLIP prompt: {clip_prompt}")
        if diffusion_prompt:
            print(f"    Diffusion prompt: {diffusion_prompt[:60]}...")
        else:
            print(f"    Diffusion prompt: None (remove task)")
        if negative_diffusion_prompt:
            print(f"    Negative prompt: {negative_diffusion_prompt[:60]}...")
        else:
            print(f"    Negative prompt: None")
        
        # Check if detected task matches selected task
        if detected_task != task.value:
            print(f"\n  Note: Agent detected '{detected_task}' but you selected '{task.value}'")
            print(f"  Proceeding with your selection: {task.value}")
        
        # For remove task: force null diffusion prompts
        if task == Task.REMOVE:
            diffusion_prompt = None
            negative_diffusion_prompt = None
        
        # Generate mask using CLIP + SAM with clip_prompt
        print(f"\n  Generating mask with CLIP + SAM...")
        print(f"    Searching for: '{clip_prompt}'")
        mask = self.generate_mask_from_text(clip_prompt)
        
        if mask is None or not np.any(mask > 0):
            print("  Failed to generate mask from text. Try manual mode.")
            return None
        
        self.mask_state.mask = mask
        print("  Mask generated successfully!")
        
        # Show mask and allow undo/apply
        cv2.namedWindow(self.window_name)
        self.show_controls(task, Mode.AGENT)
        
        while True:
            display = self.get_display_image()
            cv2.imshow(self.window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                print("  Cancelled")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('z'):  # Undo
                if self.mask_state.undo():
                    print("  Undo successful")
                else:
                    print("  Nothing to undo")
            
            elif key == ord('r'):  # Reset and retry with new clip prompt
                print("\n  Enter new CLIP prompt to search for:")
                new_clip_prompt = input("  > ").strip()
                if new_clip_prompt:
                    print(f"  Searching for: '{new_clip_prompt}'")
                    mask = self.generate_mask_from_text(new_clip_prompt)
                    if mask is not None and np.any(mask > 0):
                        self.mask_state.save_state()
                        self.mask_state.mask = mask
                        clip_prompt = new_clip_prompt
                        print("  Mask regenerated")
                    else:
                        print("  Failed to generate mask")
            
            elif key == ord('a'):  # Apply
                if np.any(self.mask_state.mask > 0):
                    print("  Applying mask...")
                    cv2.destroyAllWindows()
                    return (self.mask_state.mask, diffusion_prompt, negative_diffusion_prompt)
                else:
                    print("  No mask to apply")
        
        cv2.destroyAllWindows()
        return None
    
    # =========================================================================
    # MAIN RUN FUNCTION
    # =========================================================================
    
    def run(self, image_path: str, output_dir: str = "./feature1/results/interactive"):
        """
        Run the interactive pipeline with continuous editing support.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
        """
        import time
        
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original_image = self.image.copy()
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print("\n" + "=" * 60)
        print("  INTERACTIVE IMAGE EDITING PIPELINE")
        print("=" * 60)
        
        # Main editing loop - allows continuous editing
        while True:
            # Step 1: Select Task
            print("\n  Select Task:")
            print("    [1] Remove - Remove object from image (LAMA)")
            print("    [2] Replace - Replace object with something else (SD+LCM)")
            print("    [3] Background - Change background (SD+LCM)")
            print("    [o] Use previous Output as new input")
            print("    [r] Reset to Original image")
            print("    [q] Quit")
            
            while True:
                task_input = input("\n  Enter choice (1/2/3/o/r/q): ").strip().lower()
                if task_input == 'q':
                    print("\n  Goodbye!")
                    return
                elif task_input == 'o':
                    if hasattr(self, '_last_output') and self._last_output is not None:
                        self.image = self._last_output.copy()
                        print("  Using previous output as new input.")
                    else:
                        print("  No previous output available.")
                    continue
                elif task_input == 'r':
                    self.image = self.original_image.copy()
                    print("  Reset to original image.")
                    continue
                elif task_input == '1':
                    task = Task.REMOVE
                    break
                elif task_input == '2':
                    task = Task.REPLACE
                    break
                elif task_input == '3':
                    task = Task.BACKGROUND
                    break
                else:
                    print("  Invalid choice. Try again.")
            
            print(f"\n  Selected: {task.value.upper()}")
            
            # Step 2: Select Mode
            print("\n  Select Mode:")
            print("    [1] Manual - Box/Click/Sketch selection")
            print("    [2] Agent - Text prompt with AI assistance")
            
            while True:
                mode_input = input("\n  Enter choice (1/2): ").strip()
                if mode_input == '1':
                    mode = Mode.MANUAL
                    break
                elif mode_input == '2':
                    mode = Mode.AGENT
                    break
                else:
                    print("  Invalid choice. Try again.")
            
            print(f"\n  Selected: {mode.value.upper()} mode")
            
            # Step 3: Get mask and perform operation
            if mode == Mode.MANUAL:
                result = self.run_manual_mode(task)
            else:
                result = self.run_agent_mode(task)
            
            if result is None:
                print("\n  Operation cancelled. Returning to task selection...")
                continue
            
            mask, diffusion_prompt, negative_diffusion_prompt = result
            
            # Step 4: Perform inpainting based on task
            if task == Task.REMOVE:
                # Remove task: no diffusion prompts needed, use LAMA only
                output_image = self.perform_removal(mask)
            elif task == Task.REPLACE:
                if not diffusion_prompt:
                    print("  Error: No diffusion prompt for replacement")
                    continue
                output_image = self.perform_replacement(mask, diffusion_prompt, negative_diffusion_prompt)
            elif task == Task.BACKGROUND:
                if not diffusion_prompt:
                    print("  Error: No diffusion prompt for background fill")
                    continue
                output_image = self.perform_background_fill(mask, diffusion_prompt, negative_diffusion_prompt)
            
            # Store last output for chaining
            self._last_output = output_image.copy()
            
            # Step 5: Save and show result
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"{base_name}_{task.value}_{timestamp}.png")
            mask_path = os.path.join(output_dir, f"{base_name}_{task.value}_{timestamp}_mask.png")
            
            cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, mask)
            
            print(f"\n  Result saved to: {output_path}")
            print(f"  Mask saved to: {mask_path}")
            
            # Show comparison
            print("\n  Showing result. Press any key to continue editing...")
            h, w = self.image.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = self.image
            comparison[:, w:] = output_image
            
            cv2.imshow("Current | Result (Press any key)", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print("\n" + "-" * 60)
            print("  Edit complete! Ready for next edit...")
            print("-" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Image Editing Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="./feature1/results/interactive",
                       help="Output directory")
    parser.add_argument("--sam_ckpt", type=str, default=DEFAULT_SAM_CKPT,
                       help="Path to SAM checkpoint")
    parser.add_argument("--lama_config", type=str, 
                       default=DEFAULT_LAMA_CONFIG,
                       help="Path to LAMA config")
    parser.add_argument("--lama_ckpt", type=str, default=DEFAULT_LAMA_CKPT,
                       help="Path to LAMA checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu)")
    parser.add_argument("--preload", type=str, nargs='*', default=['sam', 'lama'],
                       help="Models to preload: sam, lama, sd, agent (default: sam lama)")
    
    args = parser.parse_args()
    
    pipeline = InteractivePipeline(
        sam_checkpoint=args.sam_ckpt,
        lama_config=args.lama_config,
        lama_checkpoint=args.lama_ckpt,
        device=args.device
    )
    
    # Preload models at startup
    if args.preload:
        pipeline.preload_models(args.preload)
    
    pipeline.run(args.image, args.output_dir)


if __name__ == "__main__":
    main()
