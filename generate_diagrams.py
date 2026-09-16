"""
generate_diagrams.py
---------------------
Generates the design diagrams required by the project rubric (system
architecture, workflow, use-case, class, sequence) as PNG images using
matplotlib, so they can be embedded in README.md, the GitHub repo docs,
and the PDF project report without any external diagramming tool.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

BOX_FACE = "#EAF2FB"
BOX_EDGE = "#2E5C8A"
ACCENT = "#C0392B"
FONT = "DejaVu Sans"


def box(ax, x, y, w, h, text, fontsize=10, face=BOX_FACE, edge=BOX_EDGE):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, family=FONT, wrap=True)
    return b


def arrow(ax, xy_from, xy_to, text=None, color="#333333", style="-|>", connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=14,
        linewidth=1.4, color=color, connectionstyle=connectionstyle,
    )
    ax.add_patch(a)
    if text:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx, my + 0.05, text, ha="center", va="bottom", fontsize=8, family=FONT, color=color)


def new_ax(figsize=(11, 7), xlim=(0, 11), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------------
# 1. System Architecture Diagram
# ----------------------------------------------------------------------
def draw_architecture():
    fig, ax = new_ax(figsize=(12, 6.5), xlim=(0, 12), ylim=(0, 7))
    ax.text(6, 6.6, "System Architecture — Document / License-Plate OCR Pipeline",
            ha="center", fontsize=13, weight="bold", family=FONT)

    # Layers
    box(ax, 0.4, 4.6, 2.3, 1.4, "Input Layer\n\nImage file(s)\n(CLI argument)", fontsize=9)

    box(ax, 3.1, 5.2, 2.6, 0.9, "Module 1\nPreprocessing &\nEnhancement", fontsize=9)
    box(ax, 3.1, 4.0, 2.6, 0.9, "Edge Detection\n(Canny, Morphology)", fontsize=9)

    box(ax, 6.1, 5.2, 2.6, 0.9, "Module 2\nRegion Localisation\n(Contours)", fontsize=9)
    box(ax, 6.1, 4.0, 2.6, 0.9, "Character\nSegmentation", fontsize=9)

    box(ax, 9.1, 5.2, 2.4, 0.9, "Module 3\nFeature Extraction\n(HOG)", fontsize=9)
    box(ax, 9.1, 4.0, 2.4, 0.9, "Classification\n(PCA + KNN)", fontsize=9)

    box(ax, 4.8, 2.0, 3.0, 1.1, "Result Assembly\n(text, confidence,\nannotated image)", fontsize=9)
    box(ax, 4.8, 0.5, 3.0, 1.0, "Output Layer\nCLI report / saved\nimage / logs", fontsize=9)

    box(ax, 0.4, 1.9, 2.3, 1.2, "Cross-cutting:\nConfig · Logging ·\nValidation\n(utils.py, config.py)", fontsize=8, face="#FCEEEC", edge=ACCENT)

    # Arrows (left to right, then down)
    arrow(ax, (2.7, 5.3), (3.1, 5.65))
    arrow(ax, (4.4, 5.2), (4.4, 4.9))
    arrow(ax, (5.7, 4.45), (6.1, 5.65))
    arrow(ax, (5.7, 4.45), (6.1, 4.45))
    arrow(ax, (8.7, 5.2), (8.7, 4.9))
    arrow(ax, (8.7, 4.45), (9.1, 5.65))
    arrow(ax, (8.7, 4.45), (9.1, 4.45))
    arrow(ax, (10.3, 4.0), (7.3, 3.1))
    arrow(ax, (6.3, 2.0), (6.3, 1.5))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_system_architecture.png"), dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. Process Flow / Workflow Diagram
# ----------------------------------------------------------------------
def draw_workflow():
    fig, ax = new_ax(figsize=(6.5, 11), xlim=(0, 6.5), ylim=(0, 15))
    ax.text(3.25, 14.6, "Process / Workflow Diagram", ha="center", fontsize=13, weight="bold", family=FONT)

    steps = [
        "Start: Load & validate\ninput image",
        "Convert to grayscale",
        "Denoise (bilateral filter) +\nCLAHE contrast enhancement",
        "Canny edge detection +\nmorphological closing",
        "Find contour candidates &\nfilter by area / aspect ratio",
        "Select best plate/document\nregion (fallback if none)",
        "Crop ROI & binarize\n(Otsu threshold)",
        "Connected-component\ncharacter segmentation",
        "Extract HOG features\nper character",
        "KNN (+PCA) classification\nof each character",
        "Assemble recognized string\n+ confidence score",
        "Annotate image & write\nlog / CLI report",
        "End",
    ]

    y = 13.6
    step_h = 0.85
    gap = 0.25
    xs = 0.8
    w = 4.9
    for i, s in enumerate(steps):
        face = "#EAF2FB" if i not in (0, len(steps) - 1) else "#DDF3E4"
        edge = BOX_EDGE if i not in (0, len(steps) - 1) else "#2E8B57"
        box(ax, xs, y, w, step_h, s, fontsize=8.3, face=face, edge=edge)
        if i < len(steps) - 1:
            arrow(ax, (xs + w / 2, y), (xs + w / 2, y - gap))
        y -= (step_h + gap)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_workflow_diagram.png"), dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Use Case Diagram
# ----------------------------------------------------------------------
def draw_use_case():
    fig, ax = new_ax(figsize=(9, 7), xlim=(0, 9), ylim=(0, 7))
    ax.text(4.5, 6.6, "Use Case Diagram", ha="center", fontsize=13, weight="bold", family=FONT)

    # Actor (stick figure, simplified)
    actor_x, actor_y = 1.0, 3.3
    ax.plot(actor_x, actor_y + 0.65, marker="o", markersize=16, color="#2E5C8A")
    ax.plot([actor_x, actor_x], [actor_y + 0.1, actor_y + 0.5], color="#2E5C8A", linewidth=2)
    ax.plot([actor_x - 0.35, actor_x + 0.35], [actor_y + 0.4, actor_y + 0.4], color="#2E5C8A", linewidth=2)
    ax.plot([actor_x, actor_x - 0.3], [actor_y + 0.1, actor_y - 0.3], color="#2E5C8A", linewidth=2)
    ax.plot([actor_x, actor_x + 0.3], [actor_y + 0.1, actor_y - 0.3], color="#2E5C8A", linewidth=2)
    ax.text(actor_x, actor_y - 0.55, "User\n(Student / Evaluator)", ha="center", fontsize=9, family=FONT)

    # System boundary
    rect = patches.FancyBboxPatch((2.6, 0.6), 5.8, 5.6, boxstyle="round,pad=0.02",
                                   linewidth=1.6, edgecolor="#555555", facecolor="none")
    ax.add_patch(rect)
    ax.text(5.5, 6.05, "OCR Pipeline System", ha="center", fontsize=10, weight="bold", family=FONT)

    use_cases = [
        ("Recognize single\nimage (CLI)", 3.0, 5.0),
        ("Batch-process a\nfolder of images", 3.0, 3.9),
        ("Train / retrain\ncharacter classifier", 3.0, 2.8),
        ("Generate synthetic\ntraining dataset", 3.0, 1.7),
        ("View annotated\noutput & logs", 6.3, 4.4),
        ("Evaluate classifier\naccuracy", 6.3, 3.0),
    ]

    for text, x, y in use_cases:
        ellipse = patches.Ellipse((x + 1.1, y), 2.2, 0.85, linewidth=1.4,
                                   edgecolor=BOX_EDGE, facecolor=BOX_FACE)
        ax.add_patch(ellipse)
        ax.text(x + 1.1, y, text, ha="center", va="center", fontsize=8, family=FONT)
        arrow(ax, (actor_x + 0.4, actor_y), (x, y), color="#888888")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_use_case_diagram.png"), dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. Class Diagram
# ----------------------------------------------------------------------
def class_box(ax, x, y, w, h, title, attrs, methods):
    box(ax, x, y, w, h, "", face="#FFFFFF")
    ax.plot([x, x + w], [y + h - 0.4, y + h - 0.4], color=BOX_EDGE, linewidth=1)
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center", fontsize=9, weight="bold", family=FONT)

    body_lines = attrs + ["— " * 6] + methods
    ytext = y + h - 0.55
    line_h = (h - 0.5) / max(len(body_lines), 1)
    for line in body_lines:
        ax.text(x + 0.1, ytext, line, ha="left", va="top", fontsize=6.6, family="monospace")
        ytext -= line_h


def draw_class_diagram():
    fig, ax = new_ax(figsize=(13, 8), xlim=(0, 13), ylim=(0, 8))
    ax.text(6.5, 7.6, "Class Diagram", ha="center", fontsize=13, weight="bold", family=FONT)

    class_box(ax, 0.3, 4.6, 3.0, 2.6, "OCRPipeline",
              ["- classifier: CharacterClassifier"],
              ["+ process_image(path): PipelineResult",
               "+ process_batch(paths): list"])

    class_box(ax, 3.8, 4.6, 2.9, 2.6, "PipelineResult",
              ["+ input_path: str", "+ recognized_text: str",
               "+ confidence: float", "+ num_characters_detected: int",
               "+ processing_time_sec: float"],
              [])

    class_box(ax, 7.2, 4.6, 2.9, 2.6, "CharacterClassifier",
              ["- knn: KNeighborsClassifier", "- pca: PCA",
               "- is_trained: bool"],
              ["+ fit(X, y): dict", "+ predict(X): tuple",
               "+ save(path)", "+ load(path)"])

    class_box(ax, 10.4, 4.6, 2.4, 2.6, "RegionCandidate",
              ["+ x, y, w, h: int", "+ area: float",
               "+ aspect_ratio: float", "+ score: float"],
              [])

    class_box(ax, 0.3, 1.4, 2.9, 2.6, "CharacterBox",
              ["+ x, y, w, h: int", "+ glyph: ndarray"],
              [])

    class_box(ax, 3.6, 1.4, 3.2, 2.6, "PlateDetector\n(module functions)",
              ["(stateless functions)"],
              ["+ find_candidate_regions()", "+ select_best_region()",
               "+ crop_region()", "+ detect_region()"])

    class_box(ax, 7.2, 1.4, 3.2, 2.6, "Segmentation\n(module functions)",
              ["(stateless functions)"],
              ["+ segment_characters(roi)"])

    class_box(ax, 10.6, 1.4, 2.2, 2.6, "FeatureExtraction\n(module functions)",
              ["(stateless functions)"],
              ["+ extract_hog_features()", "+ extract_batch()"])

    # Relationships
    arrow(ax, (3.3, 5.9), (3.8, 5.9), text="produces")
    arrow(ax, (1.8, 4.6), (1.8, 4.0), color="#555555")
    arrow(ax, (6.1, 4.6), (6.1, 4.0), color="#555555")
    arrow(ax, (9.6, 4.6), (9.6, 4.0), color="#555555")
    arrow(ax, (1.8, 5.9), (7.2, 5.9), text="uses", connectionstyle="arc3,rad=-0.3")
    ax.text(0.4, 4.35, "OCRPipeline composes/uses the modules below", fontsize=8, family=FONT, style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_class_diagram.png"), dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. Sequence Diagram
# ----------------------------------------------------------------------
def draw_sequence_diagram():
    fig, ax = new_ax(figsize=(11, 8), xlim=(0, 11), ylim=(0, 9))
    ax.text(5.5, 8.6, "Sequence Diagram — recognize(image)", ha="center", fontsize=13, weight="bold", family=FONT)

    lifelines = [
        ("User/CLI", 0.8),
        ("main.py", 2.4),
        ("OCRPipeline", 4.2),
        ("Preprocessing/\nEdge/Detector", 6.2),
        ("Segmentation", 7.8),
        ("Feature+\nClassifier", 9.4),
    ]
    top_y, bot_y = 8.0, 0.6
    for name, x in lifelines:
        box(ax, x - 0.7, top_y, 1.4, 0.5, name, fontsize=7.5)
        ax.plot([x, x], [top_y, bot_y], color="#999999", linestyle="--", linewidth=1)

    msgs = [
        (0.8, 2.4, "recognize --image", 7.3),
        (2.4, 4.2, "process_image(path)", 6.9),
        (4.2, 6.2, "load_image, preprocess,\ndetect_edges, detect_region", 6.3),
        (6.2, 4.2, "region, ROI", 5.7),
        (4.2, 7.8, "segment_characters(roi)", 5.3),
        (7.8, 4.2, "list[CharacterBox]", 4.7),
        (4.2, 9.4, "extract_batch + predict", 4.3),
        (9.4, 4.2, "labels, confidences", 3.7),
        (4.2, 2.4, "PipelineResult", 3.1),
        (2.4, 0.8, "print report /\nsave annotated image", 2.5),
    ]
    for x_from, x_to, text, y in msgs:
        color = "#333333" if x_from < x_to else "#8A2BE2"
        style = "-|>" if x_from < x_to else "-|>"
        arrow(ax, (x_from, y), (x_to, y), text=text, color=color)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_sequence_diagram.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    draw_architecture()
    draw_workflow()
    draw_use_case()
    draw_class_diagram()
    draw_sequence_diagram()
    print(f"Diagrams written to {OUT_DIR}")
