# License Plate OCR & Verification Pipeline

A modular, lightweight Computer Vision pipeline built to detect vehicle license plates and recognize registration characters using classical image processing and statistical feature classification. Runs entirely on CPU via terminal commands.

---

## Architecture Overview

```
Input Image
    │
    ▼
[Plate Detector] --> Grayscale conversion, Sobel edge filtering, contour selection
    │
    ▼
[Preprocessing] --> Adaptive thresholding, contrast normalization, skew deskewing
    │
    ▼
[Segmentation] --> Connected components, aspect-ratio gating, character slicing
    │
    ▼
[Feature Extraction] --> Spatial zoning, structural projections, directional edge histograms
    │
    ▼
[Classifier] --> Feature mapping & multi-class character prediction (SVM/k-NN)
    │
    ▼
Final String Output + Annotated Visualization
```

---

## Repository Structure

```
plate-ocr-pipeline/
├── src/
│   ├── __init__.py
│   ├── config.py               # Hyperparameters, paths, and plate geometry thresholds
│   ├── plate_detector.py       # Plate localization via edge & contour morphology
│   ├── preprocessing.py        # Denoising, contrast leveling, and binarization
│   ├── segmentation.py         # Character isolation & component filtering
│   ├── edge_features.py        # Directional gradient & edge feature extractors
│   ├── feature_extraction.py   # Normalized character embedding generation
│   ├── classifier.py           # Character inference engine
│   ├── visualizer.py           # Debug bounding-box & pipeline overlay rendering
│   └── utils.py                # File I/O, validation, and math helpers
├── main.py                     # Primary CLI entrypoint
├── requirements.txt            # Project dependencies
├── statement.md                # Problem statement & scope specification
└── README.md                   # Setup and execution guide
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.9, 3.10, or 3.11
- `pip` package manager
- Recommended: Virtual environment (`venv` or `conda`)

### 2. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/{your-username}/plate-ocr-pipeline.git
cd plate-ocr-pipeline

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Pipeline

The project is fully executable from the command line.

### Run on a Single Image

```bash
python main.py --image path/to/vehicle.jpg
```

### Run with Intermediate Visual Debugging

To inspect detected plate crops, binary masks, and segmented bounding boxes:

```bash
python main.py --image path/to/vehicle.jpg --visualize --output ./runs/output.png
```

### Run Evaluation / Batch Inference

```bash
python main.py --dir path/to/test_folder/ --save-results ./runs/results.csv
```

---

## Running Tests

Run the pipeline's unit and validation checks:

```bash
python -m unittest discover -s tests
```

(Or run individual test scripts if configured via `pytest`)

```bash
pytest tests/
```

---

## Configuration & Tuning

All geometric limits (aspect ratios, minimum/maximum plate area, character width-to-height bounds) and model thresholds are located in `src/config.py`. Adjust these constants if running on non-standard plates or varying camera resolutions.
