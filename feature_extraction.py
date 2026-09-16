"""
feature_extraction.py
----------------------
Functional Module 3 (part B): "Feature Extraction". Converts a segmented
character glyph into a fixed-length HOG (Histogram of Oriented Gradients)
feature vector, directly applying Syllabus Module 3 (Feature Extraction:
Orientation Histogram, HOG).
"""

import cv2
import numpy as np
from skimage.feature import hog

from . import config


def prepare_glyph(glyph: np.ndarray) -> np.ndarray:
    """Resize a binary glyph to a canonical size expected by the HOG extractor."""
    resized = cv2.resize(glyph, config.HOG_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def extract_hog_features(glyph: np.ndarray) -> np.ndarray:
    """Compute a HOG descriptor for a single character glyph."""
    prepared = prepare_glyph(glyph)
    features = hog(
        prepared,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_batch(glyphs) -> np.ndarray:
    """Extract HOG features for a batch/list of glyph images -> 2D feature matrix."""
    return np.array([extract_hog_features(g) for g in glyphs])
