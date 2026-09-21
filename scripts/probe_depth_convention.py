"""
Probe which direction of the Depth Anything V2 output means "deeper into the hole".

A pothole bottom is farther from the camera than the road around it. Whichever way the
depth values move inside the mask is the direction volumetric.py must treat as deep.

Run:
    python scripts/probe_depth_convention.py [image_glob] [--limit N]
"""
import sys
import glob
import argparse
import numpy as np
import cv2
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import yaml
from src.models.yolo_segmentation import PotholeSegmenter
from src.models.depth_estimation import DepthEstimator
from src.utils.weights import resolve_from_config as resolve_weights

DEFAULT_GLOB = "data/raw/indonesian_pothole/**/images/*.jpg"


def ring_mask(pothole_mask: np.ndarray, thickness: int = 25) -> np.ndarray:
    """Road surface immediately around the pothole, excluding the pothole itself."""
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated = cv2.dilate(pothole_mask.astype(np.uint8), kernel, iterations=1)
    return (dilated == 1) & (pothole_mask == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="?", default=DEFAULT_GLOB)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    with open(ROOT_DIR / "config" / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    weights = resolve_weights(
        config["models"]["yolo"],
        ("weights_path", "weights_fallback", "weights_legacy"),
    )
    segmenter = PotholeSegmenter(weights, conf_threshold=0.25)
    depth_est = DepthEstimator(config["models"]["depth"]["model_name"])

    paths = sorted(glob.glob(str(ROOT_DIR / args.pattern), recursive=True))[: args.limit]
    if not paths:
        print(f"No images matched {args.pattern}")
        return 1

    votes_higher_is_deeper = 0
    votes_lower_is_deeper = 0
    votes_tied = 0
    samples = 0
    unreadable = 0

    print(f"{'image':<42} {'road':>8} {'hole':>8} {'delta':>8}  verdict")
    print("-" * 82)

    for path in paths:
        # imread returns None for a corrupt file, and on Windows for any path holding
        # non-ASCII characters. Skipping keeps one bad file from ending the run.
        raw = cv2.imread(path)
        if raw is None:
            unreadable += 1
            continue

        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        result = segmenter.detect(image, visualize=False)
        potholes = [d for d in result["detections"] if d.class_id == 0]
        if not potholes:
            continue

        depth_map = depth_est.predict(image)
        pothole = max(potholes, key=lambda d: np.sum(d.mask))
        mask = pothole.mask

        road = ring_mask(mask)
        if not road.any() or not (mask == 1).any():
            continue

        road_median = float(np.median(depth_map[road]))
        hole_median = float(np.median(depth_map[mask == 1]))
        delta = hole_median - road_median

        if delta > 0:
            votes_higher_is_deeper += 1
            verdict = "higher = deeper"
        elif delta < 0:
            votes_lower_is_deeper += 1
            verdict = "lower = deeper"
        else:
            # No separation at all. Counting it as a vote would inflate whichever side
            # the comparison happened to fall on.
            votes_tied += 1
            verdict = "tied, no vote"

        samples += 1
        print(f"{Path(path).name[:42]:<42} {road_median:>8.4f} {hole_median:>8.4f} "
              f"{delta:>+8.4f}  {verdict}")

    if samples == 0:
        print("No pothole was detected in any sampled image.")
        return 1

    print("-" * 82)
    print(f"samples: {samples}")
    print(f"  higher value = deeper: {votes_higher_is_deeper}")
    print(f"  lower value  = deeper: {votes_lower_is_deeper}")
    if votes_tied:
        print(f"  tied, not counted:     {votes_tied}")
    if unreadable:
        print(f"  unreadable files:      {unreadable}")

    if votes_lower_is_deeper > votes_higher_is_deeper:
        print("\nVerdict: LOWER depth values are deeper (inverse depth).")
        print("volumetric.py must take the LOWEST percentile inside the pothole mask.")
    elif votes_higher_is_deeper > votes_lower_is_deeper:
        print("\nVerdict: HIGHER depth values are deeper (metric-style depth).")
        print("volumetric.py must take the HIGHEST percentile inside the pothole mask.")
    else:
        print("\nVerdict: split. The masks are probably too tight to separate hole from road.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
