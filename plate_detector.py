"""
plate_detector.py
------------------
Functional Module 2: "Region Localisation". Finds the candidate
license-plate / document-strip region in the image using contour
analysis over the edge map, filtering candidates by area and aspect
ratio (Syllabus Modules 1 & 3: transformations, contour-based
segmentation).
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from . import config
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class RegionCandidate:
    x: int
    y: int
    w: int
    h: int
    area: float
    aspect_ratio: float
    score: float


def _score_candidate(cand: RegionCandidate) -> float:
    """Heuristic score: candidates closer to a canonical plate aspect ratio
    (~3.0-4.5) and with larger area score higher."""
    ideal_ar = 3.5
    ar_penalty = abs(cand.aspect_ratio - ideal_ar)
    return cand.area / (1.0 + ar_penalty)


def find_candidate_regions(binary_or_edges: np.ndarray, image_shape) -> List[RegionCandidate]:
    """Extract rectangular candidate regions from a binary/edge image."""
    contours, _ = cv2.findContours(
        binary_or_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = image_shape[:2]
    img_area = float(img_h * img_w)

    candidates: List[RegionCandidate] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        area = w * h
        area_fraction = area / img_area
        aspect_ratio = w / float(h)

        if not (config.MIN_REGION_AREA_FRACTION <= area_fraction <= config.MAX_REGION_AREA_FRACTION):
            continue
        if not (config.MIN_REGION_ASPECT_RATIO <= aspect_ratio <= config.MAX_REGION_ASPECT_RATIO):
            continue

        cand = RegionCandidate(x, y, w, h, area, aspect_ratio, score=0.0)
        cand.score = _score_candidate(cand)
        candidates.append(cand)

    candidates.sort(key=lambda c: c.score, reverse=True)
    logger.debug("Found %d candidate regions after filtering", len(candidates))
    return candidates


def select_best_region(
    candidates: List[RegionCandidate], image_shape
) -> Optional[RegionCandidate]:
    """Pick the top-scoring candidate, falling back to a centred default
    region if nothing passed the filters (keeps the pipeline robust /
    reliable instead of crashing on difficult images)."""
    if candidates:
        return candidates[0]

    logger.warning("No region candidates passed filtering; using fallback centre crop.")
    h, w = image_shape[:2]
    fw, fh = int(w * 0.6), int(h * 0.25)
    x = (w - fw) // 2
    y = (h - fh) // 2
    return RegionCandidate(x, y, fw, fh, area=fw * fh, aspect_ratio=fw / fh, score=0.0)


def crop_region(image: np.ndarray, region: RegionCandidate, pad: int = 4) -> np.ndarray:
    """Crop a region from the image with a small safety padding."""
    h, w = image.shape[:2]
    x0 = max(region.x - pad, 0)
    y0 = max(region.y - pad, 0)
    x1 = min(region.x + region.w + pad, w)
    y1 = min(region.y + region.h + pad, h)
    return image[y0:y1, x0:x1]


def detect_region(image: np.ndarray, edges: np.ndarray) -> RegionCandidate:
    """End-to-end region detection: find + select the best candidate."""
    candidates = find_candidate_regions(edges, image.shape)
    best = select_best_region(candidates, image.shape)
    logger.info(
        "Selected region: x=%d y=%d w=%d h=%d aspect=%.2f",
        best.x, best.y, best.w, best.h, best.aspect_ratio,
    )
    return best
