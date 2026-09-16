"""
generate_stage_montage.py
---------------------------
Runs the preprocessing/edge/segmentation stages on a demo image and saves
a labelled montage figure — used as the "Screenshots / Results" section
of the project report.
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, preprocessing, edge_features, plate_detector, segmentation
from src.utils import load_image

OUT_DIR = os.path.join(config.BASE_DIR, "docs", "diagrams")


def main(image_path: str, out_name: str = "06_pipeline_stages.png"):
    image = load_image(image_path)
    stages = preprocessing.preprocess_pipeline(image)
    edges = edge_features.detect_edges(stages["enhanced"])
    edges_closed = edge_features.morphological_close(edges)
    region = plate_detector.detect_region(image, edges_closed)
    roi = plate_detector.crop_region(image, region)
    char_boxes = segmentation.segment_characters(roi)

    roi_annot = roi.copy()
    for b in char_boxes:
        cv2.rectangle(roi_annot, (b.x, b.y), (b.x + b.w, b.y + b.h), (255, 0, 0), 1)

    panels = [
        ("1. Original Input", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None),
        ("2. Grayscale", stages["gray"], "gray"),
        ("3. CLAHE Enhanced", stages["enhanced"], "gray"),
        ("4. Canny Edges (closed)", edges_closed, "gray"),
        ("5. Localised Region (ROI)", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), None),
        ("6. Segmented Characters", cv2.cvtColor(roi_annot, cv2.COLOR_BGR2RGB), None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Pipeline Stage-by-Stage Output", fontsize=14, weight="bold")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved stage montage to {out_path}")


if __name__ == "__main__":
    demo = os.path.join(config.SAMPLE_IMAGES_DIR, "plate_demo.png")
    main(demo)
