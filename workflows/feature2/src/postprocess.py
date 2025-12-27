import numpy as np
import cv2
import os

def depth_to_colormap(depth: np.ndarray, invert: bool = False):
    """Convert a single-channel depth (H,W) float array to a 3-channel BGR uint8 image for saving with cv2.
    Scales input to 0..255, applies a colormap.
    """
    if depth.ndim == 3:
        depth = depth.squeeze(0)
    d = depth.astype('float32')
    d_min, d_max = float(d.min()), float(d.max())
    if d_max - d_min < 1e-6:
        d_norm = np.zeros_like(d)
    else:
        d_norm = (d - d_min) / (d_max - d_min)
    if invert:
        d_norm = 1.0 - d_norm
    d_uint8 = (d_norm * 255.0).astype('uint8')
    colored = cv2.applyColorMap(d_uint8, cv2.COLORMAP_MAGMA)
    return colored


def save_relight_output(relit_rgb, orig_rgb, out_path):
    """
    Save relighting output as side-by-side comparison.
    
    Args:
        relit_rgb: H,W,3 float32 in [0,1] - the relit output
        orig_rgb: H,W,3 float32 in [0,1] - original input
        out_path: output file path
    """
    # Convert to uint8 BGR for cv2
    relit_bgr = cv2.cvtColor((relit_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    orig_bgr = cv2.cvtColor((orig_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Ensure same dimensions
    h = min(orig_bgr.shape[0], relit_bgr.shape[0])
    w_orig = int(orig_bgr.shape[1] * h / orig_bgr.shape[0])
    w_relit = int(relit_bgr.shape[1] * h / relit_bgr.shape[0])
    
    orig_rs = cv2.resize(orig_bgr, (w_orig, h))
    relit_rs = cv2.resize(relit_bgr, (w_relit, h))
    
    combined = cv2.hconcat([orig_rs, relit_rs])
    cv2.imwrite(out_path, combined)


def save_output(depth_tensor, orig_image_np, out_path):
    """depth_tensor: torch.Tensor or numpy array with shape (H,W) or (1,H,W)
    orig_image_np: optional original RGB image as numpy array for side-by-side visualization
    """
    import torch
    if hasattr(depth_tensor, 'cpu'):
        d = depth_tensor.detach().cpu().numpy()
    else:
        d = depth_tensor
    if d.ndim == 4:
        d = d[0,0]
    elif d.ndim == 3 and d.shape[0] in (1,):
        d = d[0]
    elif d.ndim == 3 and d.shape[0] == 3:
        # maybe HWC already
        pass
    cmap = depth_to_colormap(d)
    # write side-by-side with original (if given)
    if orig_image_np is not None:
        import cv2
        orig_bgr = cv2.cvtColor(orig_image_np, cv2.COLOR_RGB2BGR)
        h = min(orig_bgr.shape[0], cmap.shape[0])
        # resize to same height
        orig_rs = cv2.resize(orig_bgr, (int(orig_bgr.shape[1]*h/orig_bgr.shape[0]), h))
        cmap_rs = cv2.resize(cmap, (int(cmap.shape[1]*h/cmap.shape[0]), h))
        combined = cv2.hconcat([orig_rs, cmap_rs])
        cv2.imwrite(out_path, combined)
    else:
        cv2.imwrite(out_path, cmap)
