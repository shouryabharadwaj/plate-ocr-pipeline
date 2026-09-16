"""
train_classifier.py
--------------------
Loads the (synthetic) character dataset, extracts HOG features for every
glyph, trains the KNN + PCA character classifier, evaluates it on a held
-out split, and persists the trained model to disk.

Usage:
    python scripts/train_classifier.py
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.classifier import CharacterClassifier  # noqa: E402
from src.feature_extraction import extract_hog_features  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_dataset(dataset_dir: str):
    X, y = [], []
    class_dirs = sorted(
        d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)
    )
    if not class_dirs:
        raise RuntimeError(
            f"No class folders found in {dataset_dir}. "
            "Run scripts/generate_synthetic_dataset.py first."
        )

    for class_dir in class_dirs:
        label = os.path.basename(class_dir)
        image_paths = glob.glob(os.path.join(class_dir, "*.png"))
        for path in image_paths:
            glyph = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if glyph is None:
                continue
            features = extract_hog_features(glyph)
            X.append(features)
            y.append(label)

    logger.info("Loaded %d samples across %d classes", len(X), len(class_dirs))
    return np.array(X), np.array(y)


def main(dataset_dir: str, model_path: str):
    X, y = load_dataset(dataset_dir)

    clf = CharacterClassifier(use_pca=True, pca_components=40)
    report = clf.fit(X, y)

    print("\n=== Training report ===")
    print(f"Held-out test accuracy: {report['test_accuracy']*100:.2f}%")
    print(report["classification_report"])

    clf.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the character classifier.")
    parser.add_argument("--dataset-dir", type=str, default=config.CHAR_DATASET_DIR)
    parser.add_argument("--model-path", type=str, default=config.MODEL_PATH)
    args = parser.parse_args()

    main(args.dataset_dir, args.model_path)
