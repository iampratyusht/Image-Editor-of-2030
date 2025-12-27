"""
Interactive relighting inference with mouse swipe for light direction.
Swipe on the image to set light direction, use sliders to adjust parameters,
then press SPACE or click 'Apply' to run the model.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path

from .model import load_model
from .preprocess import (
    load_midas, get_depth_from_midas_bgr, smooth_depth,
    depth_to_normals_intrinsic, swipe_to_direction, physics_relight,
    make_divisible_hw
)


class InteractiveRelight:
    def __init__(self, model, midas, midas_transforms, device, intensity=1.0):
        self.model = model
        self.midas = midas
        self.midas_transforms = midas_transforms
        self.device = device
        self.intensity = intensity
        self.z_value = 0.5  # Z coordinate for light direction (-1 to 1)
        
        # Mouse state
        self.start_pt = None
        self.end_pt = None
        self.dragging = False
        self.current_pt = None
        
        # Image data (will be set when loading image)
        self.img_bgr_orig = None
        self.img_bgr = None
        self.input_rgb = None
        self.depth_s = None
        self.normals = None
        self.H0, self.W0 = 0, 0
        self.H_res, self.W_res = 0, 0
        
        # Precomputed tensors
        self.inp_chw = None
        self.normals_chw = None
        self.depth_chw = None
        
        # Result state
        self.physics_preview = None  # Quick physics preview
        self.model_result = None     # Full model inference result
        self.current_light_vec = None
        
    def load_image(self, img_path):
        """Load and preprocess image, compute depth and normals."""
        print(f"Loading image: {img_path}")
        self.img_bgr_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.img_bgr_orig is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
        
        self.H0, self.W0 = self.img_bgr_orig.shape[:2]
        
        # Resize for U-Net
        self.H_res, self.W_res = make_divisible_hw(self.H0, self.W0, divisor=8)
        self.img_bgr = cv2.resize(self.img_bgr_orig, (self.W_res, self.H_res), 
                                   interpolation=cv2.INTER_AREA)
        
        # RGB [0,1]
        self.input_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Depth from MiDaS
        print("Computing depth with MiDaS...")
        depth = get_depth_from_midas_bgr(self.img_bgr, self.midas, 
                                          self.midas_transforms, self.device)
        self.depth_s = smooth_depth(depth)
        
        # Normals from depth
        print("Computing surface normals...")
        self.normals = depth_to_normals_intrinsic(self.depth_s)
        
        # Precompute tensors that don't change
        self.inp_chw = np.transpose(self.input_rgb, (2, 0, 1))
        self.normals_chw = np.transpose(self.normals, (2, 0, 1))
        self.depth_chw = self.depth_s[None, ...]
        
        print("Image preprocessing complete!")
    
    def compute_physics_preview(self, light_vec):
        """Compute quick physics-based preview (no model inference)."""
        physics_img = physics_relight(
            input_rgb_srgb=self.input_rgb,
            depth=self.depth_s,
            normals=self.normals,
            light_vec=light_vec,
            is_point=False,
            light_pos=None,
            intensity=float(self.intensity),
        )
        
        # Resize back to original dimensions
        physics_vis = cv2.resize(physics_img, (self.W0, self.H0), interpolation=cv2.INTER_LINEAR)
        
        # Convert to BGR uint8 for display
        preview_bgr = cv2.cvtColor((physics_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return preview_bgr
        
    def compute_model_relight(self, light_vec):
        """Compute full model relighting (slower but better quality)."""
        print("Running model inference...")
        
        # Physics-based relight
        physics_img = physics_relight(
            input_rgb_srgb=self.input_rgb,
            depth=self.depth_s,
            normals=self.normals,
            light_vec=light_vec,
            is_point=False,
            light_pos=None,
            intensity=float(self.intensity),
        )
        
        # Build network input
        phys_chw = np.transpose(physics_img, (2, 0, 1))
        net_in_np = np.concatenate([self.inp_chw, phys_chw, self.normals_chw, self.depth_chw], axis=0)
        net_in = torch.from_numpy(net_in_np).unsqueeze(0).float().to(self.device)
        
        # Light conditioning
        lx, ly, lz = light_vec.tolist()
        light_cond = torch.tensor([[lx, ly, lz, float(self.intensity)]], 
                                   dtype=torch.float32).to(self.device)
        
        phys_t = torch.from_numpy(phys_chw).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            residual = self.model(net_in, light_cond)
            out = torch.clamp(phys_t + (residual - 0.5) * 0.6, 0.0, 1.0)
        
        unet_out = out[0].cpu().numpy().transpose(1, 2, 0)
        
        # Resize back to original dimensions
        unet_out_vis = cv2.resize(unet_out, (self.W0, self.H0), interpolation=cv2.INTER_LINEAR)
        
        # Convert to BGR uint8 for display
        result_bgr = cv2.cvtColor((unet_out_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        print("Model inference complete!")
        return result_bgr
    
    def get_light_vec_from_swipe(self):
        """Get light vector from current swipe points with current Z value."""
        if self.start_pt is None or self.end_pt is None:
            return None
            
        dx = self.end_pt[0] - self.start_pt[0]
        dy = self.end_pt[1] - self.start_pt[1]
        
        nx = -dx / max(self.W0, 1)
        ny = -dy / max(self.H0, 1)
        
        vec = np.array([nx, ny, -abs(self.z_value)], dtype=np.float32)
        vec /= (np.linalg.norm(vec) + 1e-9)
        return vec
    
    def draw_arrow(self, img, start, end, color=(0, 255, 255), thickness=2):
        """Draw arrow showing light direction."""
        cv2.arrowedLine(img, start, end, color, thickness, tipLength=0.3)
    
    def on_intensity_change(self, val):
        """Callback for intensity slider."""
        self.intensity = val / 10.0  # Scale: 0-30 -> 0.0-3.0
        self.model_result = None  # Invalidate model result
        self.update_preview()
        
    def on_z_change(self, val):
        """Callback for Z coordinate slider."""
        self.z_value = (val - 10) / 10.0  # Scale: 0-20 -> -1.0 to 1.0
        self.model_result = None  # Invalidate model result
        self.update_preview()
        
    def update_preview(self):
        """Update physics preview with current settings."""
        light_vec = self.get_light_vec_from_swipe()
        if light_vec is not None:
            self.current_light_vec = light_vec
            self.physics_preview = self.compute_physics_preview(light_vec)
                
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for swipe gesture."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_pt = (x, y)
            self.dragging = True
            self.current_pt = (x, y)
            self.model_result = None  # Invalidate previous model result
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.current_pt = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.end_pt = (x, y)
                self.dragging = False
                self.update_preview()
                
    def run(self, img_path, out_dir='artifacts'):
        """Run interactive relighting session."""
        self.load_image(img_path)
        
        os.makedirs(out_dir, exist_ok=True)
        
        window_name = 'Interactive Relight'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Create sliders
        # Intensity: 0-30 maps to 0.0-3.0
        cv2.createTrackbar('Intensity (x0.1)', window_name, int(self.intensity * 10), 30, self.on_intensity_change)
        # Z: 0-20 maps to -1.0 to 1.0 (slider value 10 = z value 0)
        cv2.createTrackbar('Z Depth', window_name, int((self.z_value + 1) * 10), 20, self.on_z_change)
        
        print("\n" + "="*60)
        print("INTERACTIVE RELIGHTING")
        print("="*60)
        print("Controls:")
        print("  - Click and drag to set light direction (swipe)")
        print("  - Use sliders to adjust Intensity and Z depth")
        print("  - SPACE or ENTER: Apply model (run inference)")
        print("  - 's': Save current result")
        print("  - 'r': Reset to original")
        print("  - 'q' or ESC: Quit")
        print("="*60)
        print("Yellow arrow = dragging, Blue = physics preview, Green = model result")
        print("="*60 + "\n")
        
        save_counter = 0
        
        while True:
            # Create display image
            show_img = self.img_bgr_orig.copy()
            status_text = "Swipe to set light direction"
            arrow_color = (0, 255, 255)  # Yellow default
            
            # Show model result if available
            if self.model_result is not None:
                show_img = self.model_result.copy()
                status_text = "MODEL RESULT - Press SPACE to re-apply"
                arrow_color = (0, 255, 0)  # Green for model result
            # Otherwise show physics preview
            elif self.physics_preview is not None and not self.dragging:
                show_img = self.physics_preview.copy()
                status_text = "PREVIEW - Press SPACE to apply model"
                arrow_color = (255, 100, 0)  # Blue for preview
            
            # Draw current swipe while dragging
            if self.dragging and self.start_pt and self.current_pt:
                self.draw_arrow(show_img, self.start_pt, self.current_pt, (0, 255, 255), 3)
                status_text = "Dragging... release to preview"
            # Draw completed swipe
            elif self.start_pt and self.end_pt:
                self.draw_arrow(show_img, self.start_pt, self.end_pt, arrow_color, 3)
            
            # Add info text
            info_line1 = f"Intensity: {self.intensity:.1f} | Z: {self.z_value:.1f}"
            info_line2 = status_text
            
            # Draw text with background
            cv2.rectangle(show_img, (5, 5), (450, 65), (0, 0, 0), -1)
            cv2.putText(show_img, info_line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(show_img, info_line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Light vector info
            if self.current_light_vec is not None:
                lv = self.current_light_vec
                light_info = f"Light: [{lv[0]:.2f}, {lv[1]:.2f}, {lv[2]:.2f}]"
                cv2.putText(show_img, light_info, (10, self.H0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            cv2.imshow(window_name, show_img)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                break
                
            elif key == ord(' ') or key == 13:  # SPACE or ENTER - Apply model
                if self.current_light_vec is not None:
                    self.model_result = self.compute_model_relight(self.current_light_vec)
                else:
                    print("Please swipe on the image first to set light direction!")
                    
            elif key == ord('s'):
                # Save current result
                if self.model_result is not None:
                    save_path = os.path.join(out_dir, f"{Path(img_path).stem}_model_{save_counter}.png")
                    cv2.imwrite(save_path, self.model_result)
                    print(f"Saved model result: {save_path}")
                    save_counter += 1
                elif self.physics_preview is not None:
                    save_path = os.path.join(out_dir, f"{Path(img_path).stem}_preview_{save_counter}.png")
                    cv2.imwrite(save_path, self.physics_preview)
                    print(f"Saved preview: {save_path}")
                    save_counter += 1
                else:
                    print("No result to save. Swipe on the image first!")
                    
            elif key == ord('r'):
                # Reset
                self.start_pt = None
                self.end_pt = None
                self.physics_preview = None
                self.model_result = None
                self.current_light_vec = None
                print("Reset to original")
        
        cv2.destroyAllWindows()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Interactive relighting with mouse swipe')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--out-dir', type=str, default='artifacts', help='Output directory for saved images')
    parser.add_argument('--intensity', type=float, default=1.0, help='Initial light intensity')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.weights, device=args.device)
    
    # Load MiDaS
    print("Loading MiDaS depth model...")
    midas, midas_transforms = load_midas(args.device)
    
    # Create interactive session
    session = InteractiveRelight(model, midas, midas_transforms, args.device, args.intensity)
    session.run(args.image, args.out_dir)


if __name__ == '__main__':
    main()
