import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import preprocessing


@pytest.fixture
def sample_bgr_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)


def test_to_grayscale_shape(sample_bgr_image):
    gray = preprocessing.to_grayscale(sample_bgr_image)
    assert gray.shape == (100, 200)
    assert gray.dtype == np.uint8


def test_to_grayscale_idempotent(sample_bgr_image):
    gray = preprocessing.to_grayscale(sample_bgr_image)
    gray_again = preprocessing.to_grayscale(gray)
    assert np.array_equal(gray, gray_again)


def test_denoise_preserves_shape(sample_bgr_image):
    gray = preprocessing.to_grayscale(sample_bgr_image)
    denoised = preprocessing.denoise(gray)
    assert denoised.shape == gray.shape


def test_enhance_contrast_output_range(sample_bgr_image):
    gray = preprocessing.to_grayscale(sample_bgr_image)
    enhanced = preprocessing.enhance_contrast(gray)
    assert enhanced.min() >= 0 and enhanced.max() <= 255


def test_binarize_is_binary(sample_bgr_image):
    gray = preprocessing.to_grayscale(sample_bgr_image)
    binary = preprocessing.binarize(gray)
    unique_vals = set(np.unique(binary).tolist())
    assert unique_vals.issubset({0, 255})


def test_preprocess_pipeline_keys(sample_bgr_image):
    stages = preprocessing.preprocess_pipeline(sample_bgr_image)
    assert set(stages.keys()) == {"gray", "denoised", "enhanced", "binary"}
