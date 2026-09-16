"""
utils.py
--------
Cross-cutting helpers: logging setup, safe image I/O, and input validation.
Centralising these addresses the reliability, security and
logging/monitoring non-functional requirements.
"""

import logging
import os
import sys

import cv2
import numpy as np

from . import config


class PipelineError(Exception):
    """Raised for any recoverable error inside the OCR pipeline."""


def get_logger(name: str = "ocr_pipeline") -> logging.Logger:
    """Return a module-level logger that writes to both console and a log file."""
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on repeated calls
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def validate_image_path(image_path: str) -> None:
    """Validate that a supplied path is safe and points to a supported image file."""
    if not isinstance(image_path, str) or not image_path.strip():
        raise PipelineError("Image path must be a non-empty string.")
    if not os.path.isfile(image_path):
        raise PipelineError(f"Image file not found: {image_path}")
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise PipelineError(
            f"Unsupported file extension '{ext}'. Supported: {config.SUPPORTED_EXTENSIONS}"
        )


def load_image(image_path: str) -> np.ndarray:
    """Safely load an image from disk as a BGR numpy array, with validation and downscaling."""
    validate_image_path(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError(f"OpenCV failed to decode image: {image_path}")

    h, w = image.shape[:2]
    largest_dim = max(h, w)
    if largest_dim > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / float(largest_dim)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an image to disk, creating parent directories as needed."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if not success:
        raise PipelineError(f"Failed to write output image: {output_path}")
