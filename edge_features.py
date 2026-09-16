"""
edge_features.py
-----------------
Edge-detection utilities (Syllabus Module 3: Canny, gradient-based edges,
morphological post-processing) used to expose candidate plate / document
boundaries before contour extraction.
"""

import cv2
import numpy as np

from . import config


def detect_edges(gray: np.ndarray) -> np.ndarray:
    """Canny edge detector on a (typically denoised & contrast-enhanced) grayscale image."""
    return cv2.Canny(gray, config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD)


def morphological_close(edges: np.ndarray, kernel_size=(15, 5)) -> np.ndarray:
    """Close small gaps in edge maps so that plate/document borders form
    continuous contours (rectangular kernel favours wide, plate-like shapes)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude — a lower-level edge-strength feature, included
    for report/demonstration purposes alongside the Canny operator."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(255 * magnitude / (magnitude.max() + 1e-6))
    return magnitude
