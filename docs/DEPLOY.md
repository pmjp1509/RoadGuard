# Deploying InfraSight

The app runs from one CPU-only Docker image. Hugging Face Spaces is the target these
instructions assume, but nothing in the image is specific to it, so the same build runs
on any host that can serve a container on port 7860.

## What runs where

Model weights are not in this repository and not in the image. `runs/` alone is 7.7 GB,
and even the two files the app needs come to 56 MB. They live in a Hugging Face model
repo and are downloaded on first use into the Hub cache.

| Artifact | Size | Source at runtime |
|----------|------|-------------------|
| YOLOv8m-seg pothole weights | 50 MB | Hub model repo |
| MobileNetV3 material classifier | 6 MB | Hub model repo |
| Depth Anything V2 Small | ~100 MB | Hub, `depth-anything/Depth-Anything-V2-Small-hf` |

`src/utils/weights.py` resolves each one: it uses a local file when there is one, which
keeps development on the weights under `runs/`, and falls back to the Hub otherwise.

## Measured cost

On a desktop CPU, roughly 2 s per photo end to end, and a 10 s cold load. Input size
barely matters because both models resize internally: a 640x640 image and a 3024x4032
image land within 0.1 s of each other.

Resident memory of the Python process, measured locally: 723 MB with all three models
loaded, peaking at 1210 MB while processing a 4032 px photo. A container adds its own
overhead on top of that.

The built image is 2.92 GB, of which site-packages is 1.8 GB: torch 670 MB, then polars
at 211 MB and pyarrow at 152 MB, which Streamlit pulls in whether or not the app uses
them. Nothing here is dead weight that a flag removes. Getting materially below this
would mean dropping Streamlit or splitting inference into its own image. Check the
current Spaces image size limit before assuming this fits.

The Spaces figure is a projection, not a measurement. Its CPU is slower than the machine
these numbers came from, so expect several seconds per photo rather than two.

## 1. Publish the weights

Create a model repo on Hugging Face, then upload the two files under the names
`config/config.yaml` expects:

```bash
pip install huggingface-hub
huggingface-cli login

huggingface-cli upload <username>/infrasight-weights \
    runs/detect/models/weights/multiple_pothole/multi_v1/weights/best.pt \
    yolov8m_seg_pothole.pt

huggingface-cli upload <username>/infrasight-weights \
    models/weights/road_material/material_classifier_v1.pt \
    material_classifier_v1.pt
```

Then point the config at the repo, replacing both `hub_repo` values in
`config/config.yaml`, or leave them and set `INFRASIGHT_WEIGHTS_REPO` instead.

Check that resolution works before building anything:

```bash
python -m src.utils.weights
```

It prints the path it resolved for each model and exits non-zero if either is missing.
To prove the Hub path rather than the local one, run it somewhere without `runs/`.

## 2. Create the Space

Create a Space with the Docker SDK. The frontmatter at the top of `README.md` supplies
its title, icon, and port, so the Space needs no extra configuration file.

Push this repository to the Space remote:

```bash
git remote add space https://huggingface.co/spaces/<username>/infrasight
git push space main
```

## 3. Set the environment

In the Space settings, add two variables:

| Variable | Value | Why |
|----------|-------|-----|
| `INFRASIGHT_WEIGHTS_REPO` | `<username>/infrasight-weights` | Overrides `hub_repo` without editing the committed config |
| `INFRASIGHT_EPHEMERAL` | `1` | Tells the app to warn that history does not survive a restart |

Neither is a secret. A private weights repo would also need `HF_TOKEN`, set as a secret.

## 4. Verify

The first request is slow: it downloads Depth Anything and the two weight files, about
150 MB. After that the container answers in seconds.

Worth checking by hand:

- A photo with a card produces area, depth, volume, and a cost.
- A photo without one produces the detection and depth map, and refuses to report cm.
- The PDF downloads.
- The CSV export downloads from the History page.
- The banner about disappearing history is visible in the sidebar.

## Running the image locally

```bash
docker build -t infrasight .
docker run --rm -p 7860:7860 \
    -e INFRASIGHT_WEIGHTS_REPO=<username>/infrasight-weights \
    infrasight
```

Open http://localhost:7860. The container answered its health check about 20 s after
start on the machine these numbers came from.

`.dockerignore` keeps `runs/`, `data/`, and the training code out of the build context.
Without it the build ships several gigabytes to the daemon before it starts.

One thing to leave alone: `libgl1` in the apt line. ultralytics depends on
`opencv-python`, the GUI build, and the image ends up with `cv2` 4.11.0 from that wheel.
Pinning `opencv-python-headless` alongside it does not prevent this. Both packages
install, the GUI one wins the shared `cv2` directory, and the pair wastes 154 MB on
duplicated shared libraries. Removing `libgl1` kills the container at `import cv2`.

## What this deployment does not do

**History is not durable.** SQLite at `data/infrasight.db` and the annotated images under
`data/processed/` live in the container filesystem, which Spaces resets on restart.
Persistent storage on Spaces is a paid add-on. For a demo, the PDF and CSV downloads are
the way results leave the app.

**No GPU.** The free tier is CPU only. Nothing needs changing to use a GPU host: both
`DepthEstimator` and Ultralytics pick CUDA when it is available, and `models.depth.device`
in the config is set to `auto`.

**One process, no queue.** Concurrent uploads share one Streamlit process and one copy of
each model. `app.max_batch_files` caps a single batch at 10 photos so one visitor cannot
exhaust memory, but heavy simultaneous traffic will simply be slow.

## Updating

New weights: upload to the model repo under the same filename and restart the Space. The
Hub cache is keyed by revision, so a new upload is picked up on restart.

New code: push to the Space remote. It rebuilds automatically.
