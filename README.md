---
title: InfraSight
emoji: 🕳️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# InfraSight

Measures pothole volume from a single photo. Instance segmentation finds the hole,
monocular depth estimation reads how far the surface drops into it, and a card or coin
left in frame supplies the scale that turns pixels into centimetres.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)

## What it does

- Segments potholes and the reference object with a YOLOv8-Seg model trained on both classes.
- Builds a relative depth map with Depth Anything V2 Small.
- Reports area in cm², average depth in cm, and volume in cm³.
- Renders the hole as a 3D surface in Plotly.
- Classifies the road surface as asphalt, concrete, or paving with MobileNetV3-Small.
- Estimates repair material in kg and cost in IDR.
- Reads GPS from EXIF and plots the results on a Folium map.
- Writes a PDF report per photo and exports the full history as CSV.
- Runs as a Streamlit dashboard.

## The reference object is not optional

Every figure in cm comes from an object of known size in the same photo:

| Reference | Real size | Confidence |
|-----------|-----------|------------|
| Card (ATM, KTP, SIM) | 8.5 cm × 5.4 cm | High |
| Rp500 coin | 2.7 cm diameter | Low |

A photo without one still gets its detection and its depth map, but no measurements. The
older behaviour, falling back to the pothole's own mask, made every unreferenced photo
report the same fabricated area. A blank field is the honest answer.

## Project structure

```
InfraSight/
├── config/              # config.yaml
├── data/                # Database and processed images
├── docs/                # PRD, use cases, known limitations, deployment
├── scripts/             # Utilities and archived dev scripts
├── src/
│   ├── pipeline.py      # Photo in, measurements out. No Streamlit import.
│   ├── core/            # Volumetric logic, calibration, severity, repair advisor
│   ├── models/          # YOLO, Depth Anything, material classifier
│   ├── utils/           # Logger, GPS, weight resolution
│   └── visualization/   # 3D mesh engine
├── tests/               # Assert-based scripts, no framework
└── webapp/              # Streamlit dashboard, UI only
```

`src/pipeline.py` holds everything between a JPEG and a repair estimate. It takes its
models as arguments rather than building them, so the web app, a batch script, and the
tests drive the same code. `webapp/app.py` renders and persists. It measures nothing.

## Technology stack

PyTorch 2.6, Ultralytics for YOLOv8, HuggingFace Transformers for Depth Anything, OpenCV,
Plotly, Streamlit with Streamlit-Folium, Geopy. Training data comes from RDD2022 and the
Roboflow pothole segmentation set.

## Installation

Python 3.10 or 3.11. PyTorch has no CUDA wheels for 3.14.

Training wants an NVIDIA GPU with at least 6 GB of VRAM. On an RTX 3050, 50 epochs takes 4
to 5 hours. Inference runs on CPU.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux, macOS

# Check your driver with nvidia-smi first, then pick the matching index.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# CPU only: --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

Prepare the datasets:

```bash
python -c "from src.dataset.downloader import DatasetDownloader; DatasetDownloader().download_rdd2022()"
```

Then collect 20 to 30 photos containing a card, and generate 500 or more synthetic ones.
`src/dataset/preprocessor.py` does the pasting.

Train the segmentation model:

```bash
python models/training/train_yolo.py
```

Run the app:

```bash
cd webapp
streamlit run app.py
```

`models/training/` also holds `train_material_classifier.py` for the surface classifier and
nine other scripts from earlier training runs.

## What has been measured, and what has not

| Metric | Result |
|--------|--------|
| YOLO mAP@50 | above 0.60 on the training split |
| Material classification | above 95% on the training split, never re-validated on field photos |
| Depth axis direction | 11 of 12 Indonesian pothole photos agree that smaller depth values are farther away |
| CPU latency | about 2 s per photo, 10 s cold load |
| Resident memory | 723 MB with all three models loaded, 1210 MB peak on a 4032 px photo |
| Volume accuracy | not measured |

The volume target of ±30% is a design goal, not a result. The calibration constant in
`config/config.yaml` has never been fitted against known volumes, so every depth, volume,
material quantity, and cost figure carries unknown error. Fitting it needs photos of
objects whose volume is already known, such as a measured container standing on the road
beside a reference card. `docs/PRD.md` records this as limitation L2.

## How the volume is calculated

The heuristic compares the road surface against the floor of the hole:

1. Estimate the ground plane from the reference object plus the healthy asphalt around the
   pothole, excluding any neighbouring pothole in the same bounding box.
2. Estimate the bottom from the deepest 10% of pixels inside the mask.
3. Take the signed difference. A negative result means the mask or the depth map is wrong
   for that photo, so it reports zero rather than a sign-flipped depth.
4. Multiply by the calibration constant to get centimetres.
5. Multiply the area in cm² by the depth in cm.

Depth Anything V2 emits inverse depth, so a smaller value sits farther from the camera and
the deepest pixels are the smallest ones. Re-check the direction after any change to the
depth model:

```bash
python scripts/probe_depth_convention.py --limit 12
```

## Synthetic data

The card appears in roughly 30 of the 1000 training images, which is not enough to learn
from. `src/dataset/preprocessor.py` pastes a card PNG onto RDD2022 base images at random
scale, position, and alpha, and writes the matching YOLO polygon. That moves the class
ratio from 1000:30 to 2:1.

## Tests

No framework. Each file is a script that prints its checks and exits non-zero on failure.

```bash
python tests/test_pipeline.py                 # end to end with stub models, no GPU needed
python tests/test_volumetric_calibration.py   # area, depth axis, mesh direction, unit conversion
python tests/test_severity_repair.py          # severity levels and repair costs
```

`test_pipeline.py` drives the real pipeline with stub models, so it runs in under a second
and needs no weights. If it ever needs a real model, the pipeline has grown a dependency it
should not have.

## Deployment

The app ships as a CPU-only Docker image, 2.92 GB built.

```bash
docker build -t infrasight .
docker run --rm -p 7860:7860 -e INFRASIGHT_WEIGHTS_REPO=<username>/infrasight-weights infrasight
```

Model weights are not in this repository. They are fetched from a Hugging Face model repo
at runtime by `src/utils/weights.py`, which prefers a local file when one exists.
`docs/DEPLOY.md` covers publishing the weights, creating the Space, and the limits of the
free tier.

## Documentation

`docs/PRD.md` holds the requirements, the twelve use cases, the data model, and the known
limitations. `docs/DEPLOY.md` covers deployment.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)
- [RDD2022](https://github.com/sekilab/RoadDamageDetector)
