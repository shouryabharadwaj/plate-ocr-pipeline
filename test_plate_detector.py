import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import plate_detector


def make_edge_image_with_rect(width=400, height=300, rect=(100, 120, 200, 60)):
    """Create a binary edge image containing a single filled rectangle
    with a plate-like aspect ratio."""
    img = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, thickness=-1)
    return img


def test_find_candidate_regions_detects_rectangle():
    edges = make_edge_image_with_rect()
    candidates = plate_detector.find_candidate_regions(edges, edges.shape)
    assert len(candidates) >= 1
    best = candidates[0]
    assert best.aspect_ratio > 1.5


def test_select_best_region_fallback_on_empty():
    shape = (300, 400)
    region = plate_detector.select_best_region([], shape)
    assert region is not None
    assert region.w > 0 and region.h > 0


def test_crop_region_within_bounds():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    region = plate_detector.RegionCandidate(x=50, y=50, w=100, h=40, area=4000, aspect_ratio=2.5, score=1.0)
    crop = plate_detector.crop_region(image, region, pad=5)
    assert crop.shape[0] <= 40 + 10
    assert crop.shape[1] <= 100 + 10
