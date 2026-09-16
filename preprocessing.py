"""
preprocessing.py
-----------------
Module 1 concepts applied here: image formation basics, filtering/convolution,
image enhancement and histogram processing (Syllabus Module 1).

Functions form the first functional module of the pipeline: "Image
Preprocessing & Enhancement".
"""

import cv2
import numpy as np

from . import config


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Apply bilateral filtering: smooths noise while preserving edges,
    which is important before edge-based region detection."""
    return cv2.bilateralFilter(
        gray,
        d=config.BILATERAL_D,
        sigmaColor=config.BILATERAL_SIGMA_COLOR,
        sigmaSpace=config.BILATERAL_SIGMA_SPACE,
    )


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalisation) to improve local contrast,
    demonstrating histogram-processing concepts from Module 1."""
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive thresholding to obtain a binary image robust to uneven illumination."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15,
    )


def preprocess_pipeline(image: np.ndarray) -> dict:
    """Run the full preprocessing chain and return every intermediate result,
    which is useful both for the pipeline and for generating report screenshots.
    """
    gray = to_grayscale(image)
    denoised = denoise(gray)
    enhanced = enhance_contrast(denoised)
    binary = binarize(enhanced)
    return {
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced,
        "binary": binary,
    }
