import argparse, os, glob, torch, numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
from .model import load_model, TinyRelightNet
from .preprocess import (
    load_image, load_midas, get_depth_from_midas_bgr, smooth_depth,
    depth_to_normals_intrinsic, swipe_to_direction, physics_relight,
    make_divisible_hw
)
from .postprocess import save_output, save_relight_output

def is_image_file(p):
    p = p.lower()
    return p.endswith('.jpg') or p.endswith('.jpeg') or p.endswith('.png') or p.endswith('.tif') or p.endswith('.bmp')


def run_relight_pipeline(model, img_path, device, out_path, midas=None, midas_transforms=None,
                         light_vec=None, intensity=1.0):
    """
    Full relighting pipeline matching the notebook.
    
    Args:
        model: TinyRelightNet model
        img_path: Path to input image
        device: torch device
        out_path: Output path
        midas: MiDaS model (loaded if None)
        midas_transforms: MiDaS transforms
        light_vec: Light direction vector [3], default is from top
        intensity: Light intensity scalar
    """
    # Load image
    img_bgr_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr_orig is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    
    H0, W0 = img_bgr_orig.shape[:2]
    
    # Resize to be divisible by 8 for U-Net
    H_res, W_res = make_divisible_hw(H0, W0, divisor=8)
    img_bgr = cv2.resize(img_bgr_orig, (W_res, H_res), interpolation=cv2.INTER_AREA)
    
    # RGB [0,1]
    input_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Depth from MiDaS
    depth = get_depth_from_midas_bgr(img_bgr, midas, midas_transforms, device)
    depth_s = smooth_depth(depth)
    
    # Normals from depth
    normals = depth_to_normals_intrinsic(depth_s)
    
    # Default light direction (from top)
    if light_vec is None:
        light_vec = np.array([0.0, -1.0, -1.0], dtype=np.float32)
        light_vec /= (np.linalg.norm(light_vec) + 1e-9)
    
    # Physics-based relight
    physics_img = physics_relight(
        input_rgb_srgb=input_rgb,
        depth=depth_s,
        normals=normals,
        light_vec=light_vec,
        is_point=False,
        light_pos=None,
        intensity=float(intensity),
    )
    
    # Build 10-channel network input
    inp_chw = np.transpose(input_rgb, (2, 0, 1))      # 3,H,W
    phys_chw = np.transpose(physics_img, (2, 0, 1))   # 3,H,W
    normals_chw = np.transpose(normals, (2, 0, 1))    # 3,H,W
    depth_chw = depth_s[None, ...]                     # 1,H,W
    
    net_in_np = np.concatenate([inp_chw, phys_chw, normals_chw, depth_chw], axis=0)
    net_in = torch.from_numpy(net_in_np).unsqueeze(0).float().to(device)
    
    # Light conditioning
    lx, ly, lz = light_vec.tolist()
    light_cond = torch.tensor([[lx, ly, lz, float(intensity)]], dtype=torch.float32).to(device)
    
    phys_t = torch.from_numpy(phys_chw).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        residual = model(net_in, light_cond)
        out = torch.clamp(phys_t + (residual - 0.5) * 0.6, 0.0, 1.0)
    
    unet_out = out[0].cpu().numpy().transpose(1, 2, 0)  # H,W,3
    
    # Resize back to original dimensions
    unet_out_vis = cv2.resize(unet_out, (W0, H0), interpolation=cv2.INTER_LINEAR)
    input_rgb_vis = cv2.resize(input_rgb, (W0, H0), interpolation=cv2.INTER_LINEAR)
    
    # Save output
    save_relight_output(unet_out_vis, input_rgb_vis, out_path)


def run_on_image(model, img_path, device, out_path, resize=None):
    """Legacy simple inference (depth-only output)."""
    tensor, orig_np = load_image(img_path, device=device, resize=resize)
    with torch.no_grad():
        # For legacy mode, just run the input through if model supports it
        # This won't work with TinyRelightNet which needs 10 channels + light_cond
        pred = model(tensor)
    save_output(pred, orig_np, out_path)


def main():
    parser = argparse.ArgumentParser(description='Run relighting inference with TinyRelightNet')
    parser.add_argument('--image', type=str, default=None, help='Path to an input image (or folder)')
    parser.add_argument('--input-dir', type=str, default=None, help='Alternative: input directory')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--out-dir', type=str, default='artifacts', help='Output directory')
    parser.add_argument('--out', type=str, default=None, help='Single output file path (if using --image single file)')
    parser.add_argument('--intensity', type=float, default=1.0, help='Light intensity (0.5-2.0 recommended)')
    parser.add_argument('--light-dir', type=float, nargs=3, default=None, 
                        help='Light direction vector (x y z), e.g., --light-dir 1 0 -1 for right side')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = load_model(args.weights, device=args.device)
    
    # Load MiDaS for depth estimation
    print("Loading MiDaS depth model...")
    midas, midas_transforms = load_midas(args.device)
    print("MiDaS loaded.")
    
    # Parse light direction
    light_vec = None
    if args.light_dir is not None:
        light_vec = np.array(args.light_dir, dtype=np.float32)
        light_vec /= (np.linalg.norm(light_vec) + 1e-9)
        print(f"Using light direction: {light_vec}")

    # single image
    if args.image is not None and os.path.isfile(args.image):
        out_path = args.out or os.path.join(args.out_dir, Path(args.image).stem + '_out.png')
        run_relight_pipeline(model, args.image, args.device, out_path, 
                            midas=midas, midas_transforms=midas_transforms,
                            light_vec=light_vec, intensity=args.intensity)
        print('Wrote:', out_path)
        return

    # image dir (either --image is a directory or --input-dir)
    input_dir = args.input_dir or args.image
    if input_dir is None:
        raise ValueError('Please specify --image <file|dir> or --input-dir <dir>')
    if os.path.isdir(input_dir):
        files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if is_image_file(f)]
        for f in tqdm(files, desc='Images'):
            out_path = os.path.join(args.out_dir, Path(f).stem + '_out.png')
            run_relight_pipeline(model, f, args.device, out_path,
                                midas=midas, midas_transforms=midas_transforms,
                                light_vec=light_vec, intensity=args.intensity)
    else:
        raise ValueError('Provided input is not a file or directory: ' + input_dir)

if __name__ == '__main__':
    main()
