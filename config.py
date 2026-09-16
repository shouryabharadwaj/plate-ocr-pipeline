"""
config.py
---------
Central configuration for the Document / License-Plate OCR pipeline.
Keeping tunable parameters in one place satisfies the maintainability
and configurability non-functional requirements.
"""

import os

# ----- Paths -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHAR_DATASET_DIR = os.path.join(DATA_DIR, "char_dataset")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(MODELS_DIR, "knn_classifier.pkl")
LOG_FILE = os.path.join(LOGS_DIR, "pipeline.log")

# ----- Preprocessing -----
GAUSSIAN_BLUR_KERNEL = (5, 5)
BILATERAL_D = 11
BILATERAL_SIGMA_COLOR = 17
BILATERAL_SIGMA_SPACE = 17
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ----- Edge detection -----
CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 200

# ----- Plate / region localisation -----
MIN_REGION_ASPECT_RATIO = 1.5
MAX_REGION_ASPECT_RATIO = 6.5
MIN_REGION_AREA_FRACTION = 0.01   # relative to full image area
MAX_REGION_AREA_FRACTION = 0.60
CONTOUR_APPROX_EPSILON = 0.02

# ----- Character segmentation -----
MIN_CHAR_HEIGHT_FRACTION = 0.35   # relative to plate ROI height
MAX_CHAR_HEIGHT_FRACTION = 0.98
MIN_CHAR_WIDTH_PX = 4
CHAR_PADDING_PX = 2

# ----- Feature extraction (HOG) -----
HOG_IMAGE_SIZE = (32, 32)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# ----- Classifier -----
KNN_NEIGHBOURS = 5
CLASS_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ----- Performance / robustness -----
MAX_IMAGE_DIMENSION = 1600        # images are downscaled above this, for performance
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

for _dir in (MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, SAMPLE_IMAGES_DIR, CHAR_DATASET_DIR):
    os.makedirs(_dir, exist_ok=True)
