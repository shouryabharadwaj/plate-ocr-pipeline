# License Plate OCR & Verification Pipeline

This is a lightweight computer vision pipeline that detects license plates in the vehicles and recognizes registration characters based on classical image processing and statistical feature classification. The whole pipeline is working on CPU through the terminal commands.
---
## Abstract

In this architecture, the input image is processed through several stages. First of all, the algorithm tries to detect license plates in the input image with grayscale conversion, Sobel edge filter, and appropriate contour selection.
After that, the image is processed with adaptive thresholding, contrast normalization, and skew deskewing. Next, the pipeline segments the characters with connected components, aspect-ratio gating, character slicing, etc.
Then the algorithm extracts unique features (spatial zoning, structural projections, directional edge histograms) from the characters and classifies them with a classifier that has been trained on features to predict characters using SVM/k-NN.
Finally, the pipeline returns the final string with the help of the annotated visualization.
---

## Repository Structure
```

plate-ocr-pipeline/
├── src/
│  ├── __init__.py
│  ├── config.py        # Hyperparams, paths, and geometry thresholds
│  ├── plate_detector.py    # Plate localization via edge & contour morphology
│  ├── preprocessing.py    # Denoising, contrast leveling, and binarization
│  ├── segmentation.py     # Character isolation & component filtering
│  ├── edge_features.py    # Directional gradient & edge feature extractors
│  ├── feature_extraction.py  # Normalized character embedding generation
│  ├── classifier.py      # Character inference engine
│  ├── visualizer.py      # Debug bounding-box & pipeline overlay rendering
│  └── utils.py        # File I/O, validation, and math helpers
├── main.py           # Primary CLI entrypoint
├── requirements.txt      # Project dependencies
├── statement.md        # Problem statement & scope specification
└── README.md          # Setup and execution guide
```
---
## Installation

### Requirements
-  OS: Any system compatible with Python 3.9, 3.10, or 3.11
-  python 3.9 or higher
-  pip
-  (optional) venv or conda
### Setup
Cloning the repo and setting up a virtual environment inside the repo:
```
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
Then you need to install all the dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```
---
## Usage
The project is basically used through the command line.
To run the pipeline on the image, use the following command:
```
python main.py --image path/to/vehicle.jpg
```
If you are going to use the debug mode and want to see how the pipeline processes your image and detects license plates and characters, you can use the command below:
```
python main.py --image path/to/vehicle.jpg --visualize --output ./runs/output.png
```
To run the evaluation or batch inference:
```
python main.py --dir path/to/test_folder/ --save-results ./runs/results.csv
```
---

## Tests
To run the unit tests, use the below command:
```
python -m unittest discover -s tests
```
Or
```
pytest tests/
```
---

## Configs and Tuning

All the geometric limits (aspect ratios, minimum/maximum plate area, character width-to-height bounds) and model thresholds are in the `src/config.py` file. You can tune them to get more accurate results in case of using this pipeline on some non-standard plates or resolutions.
