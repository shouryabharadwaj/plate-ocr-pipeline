"""
segmentation.py
----------------
Functional Module 3 (part A): "Character Segmentation". Splits a localised
plate / document ROI into individual character glyphs using connected-
component / contour-based segmentation (Syllabus Module 3: Image
Segmentation).
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from . import config
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class CharacterBox:
    x: int
    y: int
    w: int
    h: int
    glyph: np.ndarray  # cropped, cleaned binary glyph image


def _binarize_roi(roi_gray: np.ndarray) -> np.ndarray:
    """Otsu thresholding works well on a small, roughly-uniform ROI such as
    a cropped plate, complementing the adaptive threshold used earlier
    on the full image."""
    _, binary = cv2.threshold(
        roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def segment_characters(roi_bgr: np.ndarray) -> List[CharacterBox]:
    """Segment individual characters from a cropped plate/document ROI.

    Steps: grayscale -> Otsu binarisation -> connected components ->
    filter by relative height/width -> sort left-to-right.
    """
    if roi_bgr.size == 0:
        logger.warning("Empty ROI passed to segment_characters.")
        return []

    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
    binary = _binarize_roi(roi_gray)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    roi_h = roi_gray.shape[0]
    boxes: List[CharacterBox] = []

    for label in range(1, num_labels):  # label 0 is background
        x, y, w, h, area = stats[label]
        height_fraction = h / float(roi_h)

        if not (config.MIN_CHAR_HEIGHT_FRACTION <= height_fraction <= config.MAX_CHAR_HEIGHT_FRACTION):
            continue
        if w < config.MIN_CHAR_WIDTH_PX:
            continue

        pad = config.CHAR_PADDING_PX
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, binary.shape[1])
        y1 = min(y + h + pad, binary.shape[0])
        glyph = binary[y0:y1, x0:x1]

        boxes.append(CharacterBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0, glyph=glyph))

    boxes.sort(key=lambda b: b.x)  # left-to-right reading order
    logger.info("Segmented %d character candidates from ROI", len(boxes))
    return boxes
