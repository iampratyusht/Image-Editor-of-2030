from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch
import cv2
import math


def load_image(path, device='cpu', resize=None):
    """Load an image and return a torch tensor (1,3,H,W) normalized to [0,1].
    If resize is (W,H) tuple will resize with bilinear.
    """
    img = Image.open(path).convert('RGB')
    if resize is not None:
        img = img.resize(resize, resample=Image.BILINEAR)
    transform = T.Compose([T.ToTensor()])  # keeps values in [0,1]
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, np.array(img)


def make_divisible_hw(H, W, divisor=8):
    """Make H,W divisible by divisor for U-Net compatibility."""
    new_H = int(math.ceil(H / divisor) * divisor)
    new_W = int(math.ceil(W / divisor) * divisor)
    return new_H, new_W


def load_midas(device):
    """Load MiDaS depth estimation model."""
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    midas.to(device).eval()
    return midas, midas_transforms


def get_depth_from_midas_bgr(image_bgr, midas, midas_transforms, device):
    """Get depth map from BGR image using MiDaS."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform
    input_batch = midas_transforms(image_rgb)

    # Normalize shape → [1,3,H,W]
    if input_batch.dim() == 3:
        input_batch = input_batch.unsqueeze(0)

    input_batch = input_batch.to(device)

    with torch.no_grad():
        pred = midas(input_batch)

        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        pred = torch.nn.functional.interpolate(
            pred,
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )[0, 0]

    depth = pred.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    return depth.astype(np.float32)


def smooth_depth(depth, d=9, sigmaColor=75, sigmaSpace=75):
    """Apply bilateral filter to smooth depth map."""
    depth = depth.astype(np.float32)
    depth_s = cv2.bilateralFilter(depth, d, sigmaColor, sigmaSpace)
    return depth_s


def depth_to_normals_intrinsic(depth, fx=None, fy=None, cx=None, cy=None):
    """Convert depth to surface normals using approximate camera intrinsics."""
    depth = depth.astype(np.float32)
    H, W = depth.shape

    if fx is None:
        fx = fy = float(max(H, W))
    if cx is None:
        cx = float(W) * 0.5
    if cy is None:
        cy = float(H) * 0.5

    xs, ys = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32)
    )

    X = (xs - cx) * depth / fx
    Y = (ys - cy) * depth / fy
    Z = depth

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    Z = Z.astype(np.float32)

    dXdx = cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=3)
    dXdy = cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=3)
    dYdx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    dYdy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
    dZdx = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=3)
    dZdy = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=3)

    tx = np.stack([dXdx, dYdx, dZdx], axis=2)
    ty = np.stack([dXdy, dYdy, dZdy], axis=2)

    normals = np.cross(tx, ty)
    norms = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-9
    normals = normals / norms
    return normals.astype(np.float32)


def swipe_to_direction(start_pt, end_pt, image_shape, z_guess=1.0):
    """Convert swipe gesture to light direction vector.
    Light comes FROM the direction you swipe (inverted).
    """
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]

    H, W = image_shape
    nx = -dx / max(W, 1)  # inverted
    ny = -dy / max(H, 1)  # inverted

    vec = np.array([nx, ny, -abs(z_guess)], dtype=np.float32)
    vec /= (np.linalg.norm(vec) + 1e-9)
    return vec


def tap_to_point_light(tap_pt, depth_map, z_offset=0.0):
    """Convert tap gesture to 3D point light position."""
    x, y = int(tap_pt[0]), int(tap_pt[1])
    x = np.clip(x, 0, depth_map.shape[1] - 1)
    y = np.clip(y, 0, depth_map.shape[0] - 1)

    base_z = float(depth_map[y, x])
    return (float(x), float(y), base_z + float(z_offset))


def srgb_to_linear(img):
    """Convert sRGB to linear color space."""
    img = img.astype(np.float32)
    mask = img <= 0.04045
    lin = np.zeros_like(img, dtype=np.float32)
    lin[mask] = img[mask] / 12.92
    lin[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return lin


def linear_to_srgb(img):
    """Convert linear to sRGB color space."""
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    mask = img <= 0.0031308
    srgb = np.zeros_like(img, dtype=np.float32)
    srgb[mask] = img[mask] * 12.92
    srgb[~mask] = 1.055 * (img[~mask] ** (1.0 / 2.4)) - 0.055
    return np.clip(srgb, 0.0, 1.0)


def physics_relight(
    input_rgb_srgb,
    depth,
    normals,
    light_vec=None,
    light_color=(1.0, 1.0, 1.0),
    is_point=False,
    light_pos=None,
    kd=1.0,
    ks=0.2,
    shininess=32.0,
    ambient=0.05,
    intensity=1.0,
):
    """
    Physics-based relighting.
    
    Args:
        input_rgb_srgb: H,W,3 in [0,1] (sRGB)
        depth: H,W in [0,1]
        normals: H,W,3 unit normals
        light_vec: (3,) unit vector for directional light
        light_pos: (x,y,z) for point light (in image coord)
        intensity: scalar multiplier for (diffuse + specular)
    
    Returns:
        relit_rgb_srgb (H,W,3)
    """
    H, W, _ = input_rgb_srgb.shape
    rgb_lin = srgb_to_linear(input_rgb_srgb)
    albedo = rgb_lin

    n = normals
    light_col = np.array(light_color, dtype=np.float32).reshape(1, 1, 3)

    if is_point and light_pos is not None:
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        px = xs.astype(np.float32)
        py = ys.astype(np.float32)
        pz = depth.astype(np.float32)

        lx = light_pos[0] - px
        ly = light_pos[1] - py
        lz = light_pos[2] - pz
        L = np.stack([lx, ly, lz], axis=2)
        dist = np.linalg.norm(L, axis=2, keepdims=True) + 1e-9
        L = L / dist
        attenuation = 1.0 / (1.0 + 0.1 * dist + 0.02 * dist * dist)
    else:
        L = np.tile(light_vec.reshape(1, 1, 3).astype(np.float32), (H, W, 1))
        attenuation = np.ones((H, W, 1), dtype=np.float32)

    ndotl = np.sum(n * L, axis=2, keepdims=True)
    diff = np.clip(ndotl, 0.0, 1.0)

    v = np.array([0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 1, 3)
    h = L + v
    h = h / (np.linalg.norm(h, axis=2, keepdims=True) + 1e-9)
    spec_angle = np.clip(np.sum(n * h, axis=2, keepdims=True), 0.0, 1.0)
    spec = (spec_angle ** (shininess / 4.0))

    shadow = (ndotl > 0).astype(np.float32)

    diffuse_term = kd * diff * albedo
    spec_term = ks * spec * light_col

    out_lin = ambient * albedo + intensity * attenuation * shadow * (diffuse_term + spec_term)
    out_lin = np.clip(out_lin, 0.0, 1.0)

    out_srgb = linear_to_srgb(out_lin)
    return out_srgb.astype(np.float32)
