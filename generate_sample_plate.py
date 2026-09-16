"""
generate_sample_plate.py
--------------------------
Synthesizes demo license-plate-style images (text on a rectangular plate,
placed on a cluttered background) so the full pipeline can be exercised
end-to-end without needing a licensed / scraped real-world dataset.

Usage:
    python scripts/generate_sample_plate.py --text "KA01AB1234" --out data/sample_images/plate1.png
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


def _find_font(size: int):
    candidates = glob.glob("/usr/share/fonts/**/*Bold*.ttf", recursive=True) + \
        glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_plate_image(text: str, width: int = 640, height: int = 480, seed: int = None) -> Image.Image:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Background: textured noise to simulate a real scene
    bg = np.random.randint(80, 160, (height, width, 3), dtype=np.uint8)
    for _ in range(6):
        cv_pt1 = (random.randint(0, width), random.randint(0, height))
        cv_pt2 = (random.randint(0, width), random.randint(0, height))
        color = tuple(int(c) for c in np.random.randint(40, 200, 3))
        img_bg = Image.fromarray(bg)
        draw_bg = ImageDraw.Draw(img_bg)
        draw_bg.line([cv_pt1, cv_pt2], fill=color, width=random.randint(2, 8))
        bg = np.array(img_bg)

    image = Image.fromarray(bg).convert("RGB")

    # Plate rectangle
    plate_w, plate_h = int(width * 0.55), int(height * 0.18)
    px = (width - plate_w) // 2 + random.randint(-20, 20)
    py = (height - plate_h) // 2 + random.randint(-20, 20)

    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [px, py, px + plate_w, py + plate_h],
        fill=(240, 240, 235),
        outline=(20, 20, 20),
        width=4,
    )

    font_size = int(plate_h * 0.6)
    font = _find_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = px + (plate_w - tw) / 2 - bbox[0]
    ty = py + (plate_h - th) / 2 - bbox[1]
    draw.text((tx, ty), text, fill=(10, 10, 10), font=font)

    # Mild blur to emulate camera focus imperfection
    arr = np.array(image)
    return Image.fromarray(arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic demo plate image.")
    parser.add_argument("--text", type=str, default="KA01AB1234")
    parser.add_argument("--out", type=str, default=os.path.join(config.SAMPLE_IMAGES_DIR, "plate_demo.png"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    img = generate_plate_image(args.text, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img.save(args.out)
    print(f"Saved demo plate image to {args.out}")
