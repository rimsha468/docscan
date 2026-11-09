from dataclasses import dataclass

import cv2
import numpy as np

from .detect import find_document
from .enhance import apply_mode
from .geometry import warp_to_rect


@dataclass(frozen=True)
class ScanResult:
    image: np.ndarray
    corners: np.ndarray


def trim_border(image, fraction):
    h, w = image.shape[:2]
    dy, dx = int(h * fraction), int(w * fraction)
    return image[dy : h - dy, dx : w - dx]


def scan(image, mode="color", trim=0.01):
    """Detect, flatten and enhance a document.

    trim removes a thin strip around the warped page, where background
    pixels tend to leak in because the detected corners are never exact.
    """
    corners = find_document(image)
    flat = trim_border(warp_to_rect(image, corners), trim)
    return ScanResult(apply_mode(flat, mode), corners)


def draw_outline(image, corners, color=(0, 200, 0)):
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    thickness = max(2, max(out.shape[:2]) // 300)
    points = np.round(corners).astype(np.int32)
    cv2.polylines(out, [points], True, color, thickness, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), thickness * 3, color, -1, cv2.LINE_AA)
    return out
