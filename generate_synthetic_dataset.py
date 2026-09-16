"""
generate_synthetic_dataset.py
------------------------------
Generates a synthetic labelled dataset of character glyphs (0-9, A-Z)
by rendering each character in multiple system fonts with random
rotation, scale jitter and Gaussian noise. This removes the need for
an external internet-downloaded dataset while still producing a
realistic training set for the KNN character classifier.

Usage:
    python scripts/generate_synthetic_dataset.py --samples-per-class 120
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

CANVAS_SIZE = 64


def find_system_fonts():
    """Locate a handful of usable TrueType fonts installed on the system."""
    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    fonts = []
    for d in search_dirs:
        fonts.extend(glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True))
    if not fonts:
        logger.warning("No TTF fonts found on system; falling back to PIL default font.")
    return fonts


def render_char(char: str, font_path: str, font_size: int, angle: float, noise_sigma: float) -> np.ndarray:
    """Render a single character to a CANVAS_SIZE x CANVAS_SIZE grayscale glyph image."""
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((CANVAS_SIZE - tw) / 2 - bbox[0], (CANVAS_SIZE - th) / 2 - bbox[1])
    draw.text(pos, char, fill=255, font=font)

    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    arr = np.array(img).astype(np.float32)
    if noise_sigma > 0:
        arr += np.random.normal(0, noise_sigma, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def generate_dataset(samples_per_class: int, output_dir: str, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    fonts = find_system_fonts()
    labels = config.CLASS_LABELS

    total = 0
    for label in labels:
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            font_path = random.choice(fonts) if fonts else None
            font_size = random.randint(34, 46)
            angle = random.uniform(-12, 12)
            noise_sigma = random.uniform(0, 12)

            glyph = render_char(label, font_path, font_size, angle, noise_sigma)
            out_path = os.path.join(class_dir, f"{label}_{i:04d}.png")
            Image.fromarray(glyph).save(out_path)
            total += 1

    logger.info("Generated %d synthetic glyph images across %d classes in %s", total, len(labels), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic character dataset.")
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--output-dir", type=str, default=config.CHAR_DATASET_DIR)
    args = parser.parse_args()

    generate_dataset(args.samples_per_class, args.output_dir)
