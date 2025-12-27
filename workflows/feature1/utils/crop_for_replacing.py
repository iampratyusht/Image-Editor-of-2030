import cv2
import numpy as np
from typing import Tuple


def resize_and_pad(image: np.ndarray, mask: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    height, width, _ = image.shape
    max_dim = max(height, width)
    scale = target_size / max_dim

    new_height = int(height * scale)
    new_width = int(width * scale)

    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_height = target_size - new_height
    pad_width = target_size - new_width

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    image_padded = np.pad(image_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    mask_padded = np.pad(mask_resized, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

    return image_padded, mask_padded, (top_pad, bottom_pad, left_pad, right_pad)


def recover_size(image_padded: np.ndarray, mask_padded: np.ndarray, orig_size: Tuple[int, int],
                 padding_factors: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:

    h, w, _ = image_padded.shape
    top_pad, bottom_pad, left_pad, right_pad = padding_factors

    image = image_padded[top_pad:h-bottom_pad, left_pad:w-right_pad, :]
    mask = mask_padded[top_pad:h-bottom_pad, left_pad:w-right_pad]

    image_resized = cv2.resize(image, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, orig_size[::-1], interpolation=cv2.INTER_LINEAR)

    return image_resized, mask_resized


if __name__ == '__main__':
    image = cv2.imread('input.jpg')
    mask = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

    image_padded, mask_padded, padding_factors = resize_and_pad(image, mask)

    height, width, _ = image.shape
    image_resized, mask_resized = recover_size(image_padded, mask_padded, (height, width), padding_factors)

    cv2.imwrite('padded_image.png', image_padded)
    cv2.imwrite('padded_mask.png', mask_padded)
    cv2.imwrite('recovered_image.png', image_resized)
    cv2.imwrite('recovered_mask.png', mask_resized)
