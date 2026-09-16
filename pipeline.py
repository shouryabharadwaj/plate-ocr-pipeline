"""
pipeline.py
-----------
Orchestrates the full Document/License-Plate OCR pipeline by chaining the
three functional modules:

    1. Preprocessing & Enhancement   (preprocessing.py, edge_features.py)
    2. Region Localisation & Character Segmentation
       (plate_detector.py, segmentation.py)
    3. Feature Extraction & Recognition (feature_extraction.py, classifier.py)

Also produces an annotated output image and a structured result dict,
which is the "reporting" surface of the CLI.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import config, preprocessing, edge_features, plate_detector, segmentation
from . import feature_extraction, visualizer
from .classifier import CharacterClassifier
from .utils import get_logger, load_image, save_image, PipelineError

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    input_path: str
    recognized_text: str
    confidence: float
    num_characters_detected: int
    processing_time_sec: float
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)


class OCRPipeline:
    """High-level, reusable pipeline object. Loads the trained classifier
    once and can process any number of images (supports batch mode)."""

    def __init__(self, model_path: str = config.MODEL_PATH):
        self.classifier = CharacterClassifier.load(model_path)

    def process_image(self, image_path: str, save_annotated_to: Optional[str] = None) -> PipelineResult:
        start = time.time()
        logger.info("Processing image: %s", image_path)

        image = load_image(image_path)

        # --- Module 1: Preprocessing ---
        stages = preprocessing.preprocess_pipeline(image)
        edges = edge_features.detect_edges(stages["enhanced"])
        edges = edge_features.morphological_close(edges)

        # --- Module 2: Region localisation + character segmentation ---
        region = plate_detector.detect_region(image, edges)
        roi = plate_detector.crop_region(image, region)
        char_boxes = segmentation.segment_characters(roi)

        if not char_boxes:
            logger.warning("No characters segmented for %s", image_path)
            result = PipelineResult(
                input_path=image_path,
                recognized_text="",
                confidence=0.0,
                num_characters_detected=0,
                processing_time_sec=time.time() - start,
                annotated_image=image,
            )
            return result

        # --- Module 3: Feature extraction + classification ---
        glyphs = [box.glyph for box in char_boxes]
        features = feature_extraction.extract_batch(glyphs)
        preds, confidences = self.classifier.predict(features)

        recognized_text = "".join(preds)
        overall_confidence = float(np.mean(confidences)) if len(confidences) else 0.0

        annotated = visualizer.annotate_result(image, region, char_boxes, recognized_text, overall_confidence)

        if save_annotated_to:
            save_image(annotated, save_annotated_to)
            logger.info("Saved annotated output to %s", save_annotated_to)

        elapsed = time.time() - start
        logger.info(
            "Result for %s -> '%s' (confidence=%.2f%%, %d chars, %.3fs)",
            image_path, recognized_text, overall_confidence * 100, len(char_boxes), elapsed,
        )

        return PipelineResult(
            input_path=image_path,
            recognized_text=recognized_text,
            confidence=overall_confidence,
            num_characters_detected=len(char_boxes),
            processing_time_sec=elapsed,
            annotated_image=annotated,
        )

    def process_batch(self, image_paths: List[str], output_dir: Optional[str] = None) -> List[PipelineResult]:
        """Scalability: process a list of images sequentially, continuing on
        per-image errors instead of aborting the whole batch (reliability)."""
        results = []
        for path in image_paths:
            try:
                out_path = None
                if output_dir:
                    import os
                    out_path = os.path.join(output_dir, "annotated_" + os.path.basename(path))
                results.append(self.process_image(path, save_annotated_to=out_path))
            except PipelineError as exc:
                logger.error("Failed to process %s: %s", path, exc)
        return results
