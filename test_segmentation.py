import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import segmentation


def make_roi_with_blocks(n_chars=4, char_w=20, char_h=40, gap=10, margin=10):
    """Build a synthetic plate-like ROI with n_chars solid black rectangles
    (as stand-ins for characters) on a white background."""
    width = margin * 2 + n_chars * char_w + (n_chars - 1) * gap
    height = char_h + margin * 2
    roi = np.full((height, width, 3), 255, dtype=np.uint8)

    for i in range(n_chars):
        x0 = margin + i * (char_w + gap)
        y0 = margin
        cv2.rectangle(roi, (x0, y0), (x0 + char_w, y0 + char_h), (0, 0, 0), -1)
    return roi


def test_segment_characters_finds_correct_count():
    roi = make_roi_with_blocks(n_chars=5)
    boxes = segmentation.segment_characters(roi)
    assert len(boxes) == 5


def test_segment_characters_left_to_right_order():
    roi = make_roi_with_blocks(n_chars=4)
    boxes = segmentation.segment_characters(roi)
    xs = [b.x for b in boxes]
    assert xs == sorted(xs)


def test_segment_characters_empty_roi_returns_empty_list():
    empty_roi = np.zeros((0, 0, 3), dtype=np.uint8)
    boxes = segmentation.segment_characters(empty_roi)
    assert boxes == []


def test_segment_characters_blank_roi_returns_no_boxes():
    blank_roi = np.full((60, 200, 3), 255, dtype=np.uint8)
    boxes = segmentation.segment_characters(blank_roi)
    assert len(boxes) == 0
