# Problem Statement & Project Scope

## 1. Problem Statement

Manually entering vehicle registration numbers into parking barriers, campus toll plazas, and security checkpoints is time consuming, error-prone, and completely impractical at peak traffic. Commercial automated license plate recognition solutions require deep-learning stacks or pay for cloud-based APIs - necessitating expensive GPUs or perpetually running high-throughput internet connections.

This project proposes a self-contained, light-weight, deterministic pipeline which detects vehicle number plates and characters using classical computer vision techniques and statistical machine learning and executes fully offline on standard CPU hardware.

## 2. Project Scope

This pipeline is able to:

- Accept static RGB vehicle images via a terminal command-line interface
- Isolate the license plate region using morphological filtering and aspect-ratio thresholding
- Binarize and preprocess the extracted plate to mitigate ambient lighting and road grime
- Segment characters into isolated bounding boxes using vertical projection and connected component filtering
- Extract descriptive topological and directional edge features from glyphs
- Classify the extracted characters using a lightweight multi-class model and compile the recognized string

Not in scope:

- High-speed stitching of video frames and tracking of vehicles across consecutive CCTV frames
- Multi-line, curving, or heavily rotated non-standard custom vanity plates
- Hardware-level interfacing with boom barriers, sensors, and microcontroller triggers

## 3. Target Users

- **Campus Security & Parking Administrators**: For automatic logging of entry/exit times of student and staff vehicles without recurring cloud subscription costs
- **Toll Booth & Gated Facility Operators**: As a backup verification system or primary offline plate reader
- **Computer Vision Students & Researchers**: As a clean, modular benchmark illustrating classical feature extraction pipelines before deep learning

## 4. Key Functional Capabilities

- **Automated Localization**: Recognizes high-contrast rectangular contours in vehicle image and isolates the candidate plate area
- **Robust Character Slicing**: Extracts adjacent glyphs even with road dirt or slight baseline tilt
- **CPU-friendly Recognition**: Lightweight vector embeddings versus huge neural network weights result in fast inference on consumer laptops
- **Diagnostic Visualizer**: Shows bounding boxes, segmentation masks, and class confidence heatmaps for debugging
