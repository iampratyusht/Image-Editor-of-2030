"""
Feature 1: Object Editing Pipeline
MobileSAM + LAMA + Stable Diffusion with LCM-LoRA

Modules:
- masking: SAM-based mask generation
- inpaint_by_lama: LAMA inpainting
- inpaint_by_sd: Stable Diffusion inpainting with LCM
- clip_masking: CLIP-based text-to-mask
- object_removing: Interactive object removal
- object_replacing: Interactive object replacement
- background_filling: Interactive background replacement
- interactive_pipeline: Unified interactive pipeline
- agent_decision: AI agent for prompt parsing
"""

from .masking import predict_masks_with_sam, build_sam_model
from .inpaint_by_lama import build_lama_model, inpaint_img_with_builded_lama
from .inpaint_by_sd import (
    build_sd_inpaint_model,
    replace_img_with_sd,
    fill_img_with_sd,
    fill_background_with_sd
)

__all__ = [
    # SAM
    "predict_masks_with_sam",
    "build_sam_model",
    # LAMA
    "build_lama_model",
    "inpaint_img_with_builded_lama",
    # SD
    "build_sd_inpaint_model",
    "replace_img_with_sd",
    "fill_img_with_sd",
    "fill_background_with_sd",
]
