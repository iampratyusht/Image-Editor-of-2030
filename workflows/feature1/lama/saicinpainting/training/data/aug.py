from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
import numpy as np
import cv2


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple"""
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")
    if param is None:
        return param
    if isinstance(param, (int, float)):
        if low is None:
            low = -param if bias is None else bias - param
            param = param if bias is None else bias + param
        return (low, param)
    elif isinstance(param, (list, tuple)):
        return tuple(param)
    else:
        raise ValueError("Argument param must be either scalar or tuple/list")


class IAAAffine2(DualTransform):
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super(IAAAffine2, self).__init__(always_apply, p)
        self.scale = scale
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = shear
        self.order = order
        self.cval = cval
        self.mode = mode
        self._matrix = None

    def apply(self, img, **params):
        return self._apply_affine(img, self._matrix, self.order, self.cval, self.mode)

    def apply_to_mask(self, mask, **params):
        return self._apply_affine(mask, self._matrix, 0, 0, self.mode)

    def _apply_affine(self, img, matrix, order, cval, mode):
        if matrix is None:
            return img
        border_mode = cv2.BORDER_REFLECT if mode == "reflect" else cv2.BORDER_REPLICATE
        flags = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
        return cv2.warpAffine(img, matrix[:2], (img.shape[1], img.shape[0]),
                              flags=flags, borderMode=border_mode, borderValue=cval)

    def get_params(self):
        scale = np.random.uniform(self.scale[0], self.scale[1])
        rotate = np.random.uniform(self.rotate[0], self.rotate[1]) if self.rotate else 0
        shear_x = np.random.uniform(self.shear[0], self.shear[1]) if self.shear else 0
        shear_y = np.random.uniform(self.shear[0], self.shear[1]) if self.shear else 0
        return {"scale": scale, "rotate": rotate, "shear_x": shear_x, "shear_y": shear_y}

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        h, w = img.shape[:2]
        
        random_params = self.get_params()
        scale = random_params["scale"]
        rotate = random_params["rotate"]
        shear_x = random_params["shear_x"]
        shear_y = random_params["shear_y"]
        
        center = (w / 2, h / 2)
        
        # Create rotation matrix
        angle_rad = np.deg2rad(rotate)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Combined transformation matrix
        matrix = np.array([
            [scale * cos_a + shear_x * sin_a, -scale * sin_a + shear_x * cos_a, 0],
            [shear_y * cos_a + scale * sin_a, -shear_y * sin_a + scale * cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Translate to center and back
        T1 = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]], dtype=np.float32)
        T2 = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=np.float32)
        
        self._matrix = T2 @ matrix @ T1
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")


class IAAPerspective2(DualTransform):
    """Perform a random four point perspective transform of the input.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super(IAAPerspective2, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        self.cval = cval
        self.mode = mode
        self.order = order
        self._matrix = None
        self._output_shape = None

    def apply(self, img, **params):
        return self._apply_perspective(img, self._matrix, self._output_shape, self.order, self.cval, self.mode)

    def apply_to_mask(self, mask, **params):
        return self._apply_perspective(mask, self._matrix, self._output_shape, 0, 0, self.mode)

    def _apply_perspective(self, img, matrix, output_shape, order, cval, mode):
        if matrix is None:
            return img
        border_mode = cv2.BORDER_REPLICATE if mode == "replicate" else cv2.BORDER_REFLECT
        flags = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
        return cv2.warpPerspective(img, matrix, output_shape,
                                   flags=flags, borderMode=border_mode, borderValue=cval)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        h, w = img.shape[:2]
        
        scale = np.random.uniform(self.scale[0], self.scale[1])
        
        # Original corners
        src_points = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # Randomly perturbed corners
        dst_points = src_points + np.random.uniform(-scale * min(h, w), scale * min(h, w), (4, 2)).astype(np.float32)
        
        self._matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        if self.keep_size:
            self._output_shape = (w, h)
        else:
            # Calculate bounding box of transformed image
            corners = cv2.perspectiveTransform(src_points.reshape(1, -1, 2), self._matrix)
            x_min, y_min = corners[0].min(axis=0)
            x_max, y_max = corners[0].max(axis=0)
            self._output_shape = (int(x_max - x_min), int(y_max - y_min))
        
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("scale", "keep_size")
