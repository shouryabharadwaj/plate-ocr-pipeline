#!/usr/bin/env python3
"""
main.py
-------
Command-line entry point for the Document / License-Plate OCR Pipeline.

Examples
--------
Recognise a single image:
    python main.py recognize --image data/sample_images/plate_demo.png --output outputs/result.png

Batch process a whole folder:
    python main.py batch --input-dir data/sample_images --output-dir outputs

Train (or retrain) the classifier from the synthetic dataset:
    python main.py train
"""

import argparse
import glob
import os
import sys

from src import config
from src.utils import get_logger, PipelineError

logger = get_logger(__name__)


def cmd_recognize(args):
    from src.pipeline import OCRPipeline

    pipeline = OCRPipeline(model_path=args.model)
    result = pipeline.process_image(args.image, save_annotated_to=args.output)

    print("\n=== OCR Pipeline Result ===")
    print(f"Input image        : {result.input_path}")
    print(f"Recognized text     : {result.recognized_text or '(none detected)'}")
    print(f"Confidence          : {result.confidence * 100:.2f}%")
    print(f"Characters detected : {result.num_characters_detected}")
    print(f"Processing time     : {result.processing_time_sec:.3f}s")
    if args.output:
        print(f"Annotated image     : {args.output}")


def cmd_batch(args):
    from src.pipeline import OCRPipeline

    image_paths = []
    for ext in config.SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No supported images found in {args.input_dir}")
        return

    pipeline = OCRPipeline(model_path=args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    results = pipeline.process_batch(image_paths, output_dir=args.output_dir)

    print(f"\n=== Batch Result ({len(results)}/{len(image_paths)} succeeded) ===")
    for r in results:
        print(f"{os.path.basename(r.input_path):30s} -> {r.recognized_text:15s} "
              f"({r.confidence*100:5.1f}%, {r.num_characters_detected} chars, {r.processing_time_sec:.3f}s)")


def cmd_train(args):
    # Lazy imports so `python main.py --help` stays fast
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scripts.generate_synthetic_dataset import generate_dataset
    from scripts.train_classifier import main as train_main

    has_class_dirs = any(
        os.path.isdir(os.path.join(config.CHAR_DATASET_DIR, entry))
        for entry in os.listdir(config.CHAR_DATASET_DIR)
    )
    if args.regenerate_data or not has_class_dirs:
        print("Generating synthetic character dataset ...")
        generate_dataset(args.samples_per_class, config.CHAR_DATASET_DIR)

    print("Training classifier ...")
    train_main(config.CHAR_DATASET_DIR, config.MODEL_PATH)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ocr-pipeline",
        description="Document / License-Plate OCR Pipeline (CSE3010 Computer Vision project)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_recognize = subparsers.add_parser("recognize", help="Run OCR on a single image")
    p_recognize.add_argument("--image", required=True, help="Path to input image")
    p_recognize.add_argument("--output", default=None, help="Path to save annotated output image")
    p_recognize.add_argument("--model", default=config.MODEL_PATH, help="Path to trained classifier")
    p_recognize.set_defaults(func=cmd_recognize)

    p_batch = subparsers.add_parser("batch", help="Run OCR on every image in a folder")
    p_batch.add_argument("--input-dir", required=True, help="Folder of input images")
    p_batch.add_argument("--output-dir", default=config.OUTPUTS_DIR, help="Folder to save annotated outputs")
    p_batch.add_argument("--model", default=config.MODEL_PATH, help="Path to trained classifier")
    p_batch.set_defaults(func=cmd_batch)

    p_train = subparsers.add_parser("train", help="Generate synthetic data (if needed) and train the classifier")
    p_train.add_argument("--samples-per-class", type=int, default=120)
    p_train.add_argument("--regenerate-data", action="store_true", help="Force regeneration of synthetic dataset")
    p_train.set_defaults(func=cmd_train)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except PipelineError as exc:
        logger.error(str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
