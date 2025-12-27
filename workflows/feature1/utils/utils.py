import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def erode_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.erode(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def show_mask(ax, mask: np.ndarray, random_color=False):
    mask = mask.astype(np.uint8)
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)

def _get_display_size(img_shape, max_width=1920, max_height=1080):
    """Calculate optimal display size to fit screen while maintaining aspect ratio."""
    h, w = img_shape[:2]
    
    # Use 80% of screen size for safety
    max_w = int(max_width * 0.8)
    max_h = int(max_height * 0.8)
    
    # Calculate scaling factor
    scale_w = max_w / w if w > max_w else 1
    scale_h = max_h / h if h > max_h else 1
    scale = min(scale_w, scale_h, 1)  # Don't upscale
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return new_w, new_h, scale


def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    
    # Calculate display size
    display_w, display_h, scale = _get_display_size(img.shape)
    
    # Resize for display if needed
    if scale < 1:
        display_img = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_AREA)
    else:
        display_img = img.copy()
    
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", display_w, display_h)
    cv2.imshow("image", display_img)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, display_img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(display_img, tuple(last_point), max(3, int(5 * scale)), (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(display_img, tuple(last_point), max(3, int(5 * scale)), (0, 0, 255), -1)
            cv2.imshow("image", display_img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Convert display coordinates back to original image coordinates
    if last_point and scale < 1:
        last_point = [int(last_point[0] / scale), int(last_point[1] / scale)]

    return last_point


def get_bounding_box(img_path):
    """
    Interactive bounding box selection using mouse drag.
    Click and drag to draw a rectangle around the object.
    Press 'r' to reset, 'q' or ESC to confirm selection.
    """
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    
    # Calculate display size
    display_w, display_h, scale = _get_display_size(img.shape)
    
    # Resize for display if needed
    if scale < 1:
        display_img = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_AREA)
        display_img_copy = display_img.copy()
    else:
        display_img = img.copy()
        display_img_copy = img.copy()
    
    cv2.namedWindow("Select Bounding Box", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Bounding Box", display_w, display_h)
    
    # Variables for bounding box
    start_point = None
    end_point = None
    drawing = False
    bbox = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal start_point, end_point, drawing, display_img, display_img_copy, bbox
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # Update end point and redraw
                display_img = display_img_copy.copy()
                end_point = (x, y)
                cv2.rectangle(display_img, start_point, end_point, (0, 255, 0), max(1, int(2 * scale)))
                cv2.imshow("Select Bounding Box", display_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            drawing = False
            end_point = (x, y)
            cv2.rectangle(display_img, start_point, end_point, (0, 255, 0), max(1, int(2 * scale)))
            cv2.imshow("Select Bounding Box", display_img)
            
            # Store bounding box [x1, y1, x2, y2]
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])
            bbox = [x1, y1, x2, y2]
    
    cv2.setMouseCallback("Select Bounding Box", mouse_callback)
    cv2.imshow("Select Bounding Box", display_img)
    
    print("\nBounding Box Selection:")
    print("  - Click and drag to draw a rectangle around the object")
    print("  - Press 'r' to reset")
    print("  - Press 'q' or ESC to confirm selection\n")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            # Reset
            display_img = display_img_copy.copy()
            start_point = None
            end_point = None
            bbox = []
            cv2.imshow("Select Bounding Box", display_img)
            print("Selection reset")
            
        elif key == ord('q') or key == 27:  # 'q' or ESC
            # Confirm and exit
            break
    
    cv2.destroyAllWindows()
    
    if not bbox or len(bbox) != 4:
        print("Warning: No valid bounding box selected!")
        return None
    
    # Convert display coordinates back to original image coordinates
    if scale < 1:
        bbox = [int(coord / scale) for coord in bbox]
    
    print(f"Selected bounding box: {bbox}")
    return bbox


def get_sketch_mask(img_path):
    """
    Get a mask by sketching/drawing on the image.
    User draws directly on areas to segment.
    
    Returns:
        mask: Binary mask (numpy array) where drawn areas are white (255)
    """
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    
    # Calculate display size
    display_w, display_h, scale = _get_display_size(img.shape)
    
    # Resize for display if needed
    if scale < 1:
        display_img = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_AREA)
    else:
        display_img = img.copy()
    
    # Create mask for drawing
    mask = np.zeros((display_h, display_w), dtype=np.uint8)
    drawing = False
    brush_size = 15
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, mask, display_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, (x, y), brush_size, 255, -1)
            # Show the drawing on the display image
            overlay = display_img.copy()
            cv2.circle(overlay, (x, y), brush_size, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_img, 0.5, 0, display_img)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(mask, (x, y), brush_size, 255, -1)
                overlay = display_img.copy()
                cv2.circle(overlay, (x, y), brush_size, (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.5, display_img, 0.5, 0, display_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    cv2.namedWindow("Sketch Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sketch Mask", display_w, display_h)
    cv2.setMouseCallback("Sketch Mask", mouse_callback)
    
    print("\nSketch Selection:")
    print("  - Click and drag to draw/sketch on the object you want to segment")
    print("  - Press '+' or '=' to increase brush size")
    print("  - Press '-' to decrease brush size")
    print("  - Press 'c' to clear and start over")
    print("  - Press 'q' or ESC to confirm selection\n")
    print(f"Current brush size: {brush_size}")
    
    # Keep a copy for reset
    img_copy = display_img.copy()
    
    while True:
        cv2.imshow("Sketch Mask", display_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('c'):  # Clear
            mask = np.zeros((display_h, display_w), dtype=np.uint8)
            if scale < 1:
                display_img = cv2.resize(cv2.imread(img_path), (display_w, display_h), interpolation=cv2.INTER_AREA)
            else:
                display_img = cv2.imread(img_path).copy()
            img_copy = display_img.copy()
            print("Sketch cleared")
        elif key == ord('+') or key == ord('='):  # Increase brush size
            brush_size = min(50, brush_size + 5)
            print(f"Brush size: {brush_size}")
        elif key == ord('-'):  # Decrease brush size
            brush_size = max(5, brush_size - 5)
            print(f"Brush size: {brush_size}")
    
    cv2.destroyAllWindows()
    
    # Scale mask back to original size if needed
    if scale < 1:
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Check if mask has any content
    if np.max(mask) == 0:
        print("Warning: No sketch drawn!")
        return None
    
    print(f"Sketch mask created with {np.sum(mask > 0)} pixels")
    return mask


def get_interactive_selection(img_path, selection_mode='point'):
    """
    Get interactive selection (point, box, or sketch) from user.
    
    Args:
        img_path: Path to the image
        selection_mode: 'point' for single point, 'box' for bounding box, 'sketch' for free-form drawing
        
    Returns:
        For point mode: [x, y]
        For box mode: [x1, y1, x2, y2]
        For sketch mode: mask (numpy array)
    """
    if selection_mode == 'point':
        print("\n=== Point Selection Mode ===")
        print("Instructions:")
        print("  - Left click on the object center")
        print("  - Right click to confirm\n")
        return get_clicked_point(img_path)
    elif selection_mode == 'box':
        print("\n=== Bounding Box Selection Mode ===")
        return get_bounding_box(img_path)
    elif selection_mode == 'sketch':
        print("\n=== Sketch Selection Mode ===")
        return get_sketch_mask(img_path)
    else:
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Use 'point', 'box', or 'sketch'")


# =============================================================================
# Visualization Helper Functions for Object Removal Pipeline
# =============================================================================

def create_mask_overlay(img: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """
    Create an overlay visualization of mask on image.
    
    Args:
        img: Original image (H, W, 3) RGB
        mask: Binary mask (H, W) where non-zero = masked region
        color: RGB color for mask overlay (default: red)
        alpha: Transparency of overlay (0-1)
    
    Returns:
        Overlay image with mask highlighted
    """
    overlay = img.copy()
    mask_colored = np.zeros_like(img)
    mask_bool = mask > 0
    mask_colored[mask_bool] = color
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)
    return overlay


def create_masked_image(img: np.ndarray, mask: np.ndarray, fill_color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Create image with masked region filled with a solid color.
    
    Args:
        img: Original image (H, W, 3) RGB
        mask: Binary mask (H, W) where non-zero = masked region
        fill_color: RGB color to fill masked region (default: white)
    
    Returns:
        Image with masked region filled
    """
    result = img.copy()
    mask_bool = mask > 0
    result[mask_bool] = fill_color
    return result


def create_side_by_side(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray = None, 
                        labels: List[str] = None, padding: int = 10) -> np.ndarray:
    """
    Create side-by-side comparison of 2 or 3 images.
    
    Args:
        img1, img2, img3: Images to compare (must have same height)
        labels: Optional labels for each image
        padding: Padding between images
    
    Returns:
        Combined side-by-side image
    """
    images = [img1, img2]
    if img3 is not None:
        images.append(img3)
    
    # Ensure all images have same height
    max_h = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        if img.shape[0] != max_h:
            scale = max_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, max_h))
        resized.append(img)
    
    # Add padding
    pad = np.ones((max_h, padding, 3), dtype=np.uint8) * 128
    
    result = resized[0]
    for img in resized[1:]:
        result = np.hstack([result, pad, img])
    
    # Add labels if provided
    if labels:
        label_h = 30
        labeled = np.ones((max_h + label_h, result.shape[1], 3), dtype=np.uint8) * 255
        labeled[label_h:, :] = result
        
        x_offset = 0
        for i, (img, label) in enumerate(zip(resized, labels)):
            text_x = x_offset + img.shape[1] // 2 - len(label) * 5
            cv2.putText(labeled, label, (text_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 0), 2)
            x_offset += img.shape[1] + padding
        result = labeled
    
    return result


def create_comparison_grid(original: np.ndarray, mask: np.ndarray, 
                           inpainted: np.ndarray, mask_overlay: np.ndarray = None) -> np.ndarray:
    """
    Create a 2x2 grid comparison of the inpainting pipeline results.
    
    Args:
        original: Original input image
        mask: Binary mask
        inpainted: Inpainted result
        mask_overlay: Optional mask overlay (will be created if None)
    
    Returns:
        2x2 grid image showing: Original | Mask Overlay | Mask | Inpainted
    """
    h, w = original.shape[:2]
    
    # Create mask overlay if not provided
    if mask_overlay is None:
        mask_overlay = create_mask_overlay(original, mask)
    
    # Convert mask to 3-channel for display
    if mask.ndim == 2:
        mask_rgb = np.stack([mask, mask, mask], axis=2)
    else:
        mask_rgb = mask
    
    # Ensure all images are same size
    def resize_to(img, target_h, target_w):
        if img.shape[0] != target_h or img.shape[1] != target_w:
            return cv2.resize(img, (target_w, target_h))
        return img
    
    original = resize_to(original, h, w)
    mask_overlay = resize_to(mask_overlay, h, w)
    mask_rgb = resize_to(mask_rgb, h, w)
    inpainted = resize_to(inpainted, h, w)
    
    # Create grid with labels
    pad = 5
    label_h = 25
    
    # Top row: Original | Mask Overlay
    top_row = np.hstack([original, np.ones((h, pad, 3), dtype=np.uint8) * 200, mask_overlay])
    
    # Bottom row: Mask | Inpainted
    bottom_row = np.hstack([mask_rgb, np.ones((h, pad, 3), dtype=np.uint8) * 200, inpainted])
    
    # Combine rows
    grid = np.vstack([top_row, np.ones((pad, top_row.shape[1], 3), dtype=np.uint8) * 200, bottom_row])
    
    return grid


def show_results_cv2(original: np.ndarray, mask: np.ndarray, inpainted: np.ndarray,
                     window_name: str = "Results", wait: bool = True) -> None:
    """
    Display results using OpenCV windows.
    
    Args:
        original: Original image (RGB)
        mask: Binary mask
        inpainted: Inpainted result (RGB)
        window_name: Name for the display window
        wait: Whether to wait for key press
    """
    # Create mask overlay
    mask_overlay = create_mask_overlay(original, mask)
    
    # Create comparison grid
    grid = create_comparison_grid(original, mask, inpainted, mask_overlay)
    
    # Convert RGB to BGR for OpenCV display
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    
    # Display
    cv2.imshow(window_name, grid_bgr)
    if wait:
        cv2.waitKey(0)


def save_all_results(original: np.ndarray, mask: np.ndarray, inpainted: np.ndarray,
                     output_dir: str, prefix: str, timestamp: int = None) -> Dict[str, str]:
    """
    Save all results from the inpainting pipeline.
    
    Args:
        original: Original input image (RGB)
        mask: Binary mask
        inpainted: Inpainted result (RGB)
        output_dir: Directory to save results
        prefix: Filename prefix (usually input image name)
        timestamp: Optional timestamp for unique filenames
    
    Returns:
        Dictionary with paths to all saved files
    """
    import os
    import time
    
    if timestamp is None:
        timestamp = int(time.time())
    
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # 1. Save mask
    mask_path = os.path.join(output_dir, f"{prefix}_mask_{timestamp}.png")
    cv2.imwrite(mask_path, mask)
    paths['mask'] = mask_path
    
    # 2. Save mask overlay
    mask_overlay = create_mask_overlay(original, mask)
    overlay_path = os.path.join(output_dir, f"{prefix}_mask_overlay_{timestamp}.png")
    save_array_to_img(mask_overlay, overlay_path)
    paths['mask_overlay'] = overlay_path
    
    # 3. Save masked image (original with mask region highlighted)
    masked_img = create_masked_image(original, mask, fill_color=(0, 0, 0))
    masked_path = os.path.join(output_dir, f"{prefix}_masked_{timestamp}.png")
    save_array_to_img(masked_img, masked_path)
    paths['masked'] = masked_path
    
    # 4. Save inpainted result
    inpainted_path = os.path.join(output_dir, f"{prefix}_inpainted_{timestamp}.png")
    save_array_to_img(inpainted, inpainted_path)
    paths['inpainted'] = inpainted_path
    
    # 5. Save comparison grid
    grid = create_comparison_grid(original, mask, inpainted, mask_overlay)
    comparison_path = os.path.join(output_dir, f"{prefix}_comparison_{timestamp}.png")
    save_array_to_img(grid, comparison_path)
    paths['comparison'] = comparison_path
    
    # 6. Save side-by-side (original vs inpainted)
    side_by_side = create_side_by_side(original, inpainted, labels=['Original', 'Inpainted'])
    sidebyside_path = os.path.join(output_dir, f"{prefix}_before_after_{timestamp}.png")
    save_array_to_img(side_by_side, sidebyside_path)
    paths['before_after'] = sidebyside_path
    
    return paths


def save_minimal_results(original: np.ndarray, mask: np.ndarray, inpainted: np.ndarray,
                         output_dir: str, prefix: str, timestamp: int = None) -> Dict[str, str]:
    """
    Save minimal results (mask, masked overlay, inpainted) - for CLIP/text mode.
    
    Args:
        original: Original input image (RGB)
        mask: Binary mask
        inpainted: Inpainted result (RGB)
        output_dir: Directory to save results
        prefix: Filename prefix
        timestamp: Optional timestamp
    
    Returns:
        Dictionary with paths to saved files
    """
    import os
    import time
    
    if timestamp is None:
        timestamp = int(time.time())
    
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # 1. Save final mask
    mask_path = os.path.join(output_dir, f"{prefix}_mask_{timestamp}.png")
    cv2.imwrite(mask_path, mask)
    paths['mask'] = mask_path
    
    # 2. Save mask overlay (masked image visualization)
    mask_overlay = create_mask_overlay(original, mask)
    overlay_path = os.path.join(output_dir, f"{prefix}_masked_{timestamp}.png")
    save_array_to_img(mask_overlay, overlay_path)
    paths['masked'] = overlay_path
    
    # 3. Save inpainted result
    inpainted_path = os.path.join(output_dir, f"{prefix}_inpainted_{timestamp}.png")
    save_array_to_img(inpainted, inpainted_path)
    paths['inpainted'] = inpainted_path
    
    return paths


def print_saved_results(paths: Dict[str, str], title: str = "Results saved to:") -> None:
    """
    Print paths of saved results in a formatted way.
    
    Args:
        paths: Dictionary of result type -> file path
        title: Title to print before the list
    """
    print(f"\n  {title}")
    for result_type, path in paths.items():
        print(f"    {result_type.capitalize()}: {path}")