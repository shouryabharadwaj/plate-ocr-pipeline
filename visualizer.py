"""
visualizer.py
-------------
Draws detected regions, segmented character boxes, and recognised text
onto the original image for output/reporting purposes.
"""

from typing import List

import cv2
import numpy as np

from .plate_detector import RegionCandidate
from .segmentation import CharacterBox


def annotate_result(
    image: np.ndarray,
    region: RegionCandidate,
    char_boxes: List[CharacterBox],
    recognized_text: str,
    confidence: float,
) -> np.ndarray:
    """Return a copy of the image with the detected region, character boxes,
    and the final recognised text overlaid."""
    output = image.copy()

    # Region rectangle
    cv2.rectangle(
        output,
        (region.x, region.y),
        (region.x + region.w, region.y + region.h),
        (0, 255, 0),
        2,
    )

    # Character boxes (offset into the region's coordinate frame)
    for box in char_boxes:
        cv2.rectangle(
            output,
            (region.x + box.x, region.y + box.y),
            (region.x + box.x + box.w, region.y + box.y + box.h),
            (255, 0, 0),
            1,
        )

    label = f"{recognized_text}  ({confidence*100:.1f}%)"
    text_y = max(region.y - 10, 20)
    cv2.putText(
        output,
        label,
        (region.x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output
