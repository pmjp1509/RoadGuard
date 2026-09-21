"""Verification test for the analysis pipeline.

Exercises `Pipeline.analyze` through its own interface with stub models, so it runs in
under a second with no GPU, no weights, and no Streamlit. The stubs are what make the
dependency injection worth having: if these ever need a real model, the pipeline has
grown a dependency it should not have.

Run: python tests/test_pipeline.py
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline, AnalysisResult

ROAD_DEPTH = 0.8  # inverse depth of flat road: larger means nearer the camera
IMAGE_SIZE = 200

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))


# ── Stubs ─────────────────────────────────────────────────────────────────────
@dataclass
class StubDetection:
    class_id: int
    mask: np.ndarray
    bbox: tuple


class StubSegmenter:
    """Returns whatever detections the test hands it."""

    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, visualize=True):
        return {
            "detections": self.detections,
            "annotated_image": np.zeros_like(image),
        }

    @staticmethod
    def get_largest_detection(detections, class_id):
        matching = [d for d in detections if d.class_id == class_id]
        if not matching:
            return None
        return max(matching, key=lambda d: np.sum(d.mask))


class StubDepthEstimator:
    def __init__(self, depth_map):
        self.depth_map = depth_map

    def predict(self, image):
        return self.depth_map

    @staticmethod
    def visualize_depth(depth_map):
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)


class StubMaterialClassifier:
    def __init__(self, class_name, confidence):
        self.prediction = {"class": class_name, "confidence": confidence}

    def predict(self, crop):
        return self.prediction


CONFIG = {
    "volumetric": {"calibration_constant": 10.0},
    "models": {"material": {"confidence_threshold": 0.6}},
}


def scene(n_potholes=1, with_reference=True):
    """A flat road with n potholes and, optionally, a card beside them.

    The reference covers 459 px against the card's real 45.9 cm², giving exactly 10
    pixels per cm². Each pothole covers 1000 px, so 100 cm², sunk 0.3 below the road,
    which the calibration constant of 10.0 turns into 3 cm and 300 cm³.
    """
    depth = np.full((IMAGE_SIZE, IMAGE_SIZE), ROAD_DEPTH, dtype=np.float32)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    detections = []

    for index in range(n_potholes):
        top = 20 + index * 45
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask[top:top + 40, 100:125] = 1  # 40 x 25 = 1000 px
        depth[mask == 1] = ROAD_DEPTH - 0.3
        detections.append(StubDetection(0, mask, (100, top, 125, top + 40)))

    if with_reference:
        ref = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        ref[10:27, 10:37] = 1  # 17 x 27 = 459 px, aspect ratio 1.59 so it reads as a card
        detections.append(StubDetection(1, ref, (10, 10, 37, 27)))

    return image, depth, detections


def build(n_potholes=1, with_reference=True, material=None):
    image, depth, detections = scene(n_potholes, with_reference)
    pipeline = Pipeline(
        StubSegmenter(detections), StubDepthEstimator(depth), material, CONFIG
    )
    return pipeline.analyze(image)


# ── No pothole ────────────────────────────────────────────────────────────────
empty = Pipeline(
    StubSegmenter([]),
    StubDepthEstimator(np.full((IMAGE_SIZE, IMAGE_SIZE), ROAD_DEPTH, dtype=np.float32)),
    None,
    CONFIG,
).analyze(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

check("no pothole in the photo returns None", empty is None, f"got {type(empty)}")

# ── No reference object ───────────────────────────────────────────────────────
# The bug this guards: the pipeline used to fall back to the pothole's own mask as the
# reference, which made every unscaled photo report exactly 45.9 cm².
unscaled = build(n_potholes=2, with_reference=False)

check("a photo without a reference object is marked unscaled",
      unscaled is not None and not unscaled.scaled)
check("an unscaled result carries no measurements", unscaled.potholes == [])
check("an unscaled result carries no summary", unscaled.summary is None)
check("an unscaled result still counts the potholes it found",
      unscaled.pothole_count == 2, f"got {unscaled.pothole_count}")
check("an unscaled result still carries the images to show",
      unscaled.annotated is not None and unscaled.depth_viz is not None)

# A reference detected but too small to cover a pixel has no area to calibrate against.
# Dividing by it used to raise out of analyze and drop the whole photo.
image, depth, detections = scene(1, with_reference=False)
detections.append(
    StubDetection(1, np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8), (10, 10, 12, 12))
)
try:
    empty_reference = Pipeline(
        StubSegmenter(detections), StubDepthEstimator(depth), None, CONFIG
    ).analyze(image)
    check("an empty reference mask is treated as no reference, not an error",
          empty_reference is not None and not empty_reference.scaled)
except Exception as exc:
    check("an empty reference mask is treated as no reference, not an error",
          False, f"raised {type(exc).__name__}: {exc}")

# ── One pothole, measured ─────────────────────────────────────────────────────
single = build(n_potholes=1)

check("a measured result is marked scaled", single.scaled)
check("one pothole in, one measurement out", len(single.potholes) == 1)
check("area is 100 cm²", abs(single.summary.area_cm2 - 100.0) < 0.01,
      f"got {single.summary.area_cm2:.4f}")
check("depth is 3 cm", abs(single.summary.avg_depth_cm - 3.0) < 0.01,
      f"got {single.summary.avg_depth_cm:.4f}")
check("volume is 300 cm³", abs(single.summary.volume_cm3 - 300.0) < 0.1,
      f"got {single.summary.volume_cm3:.4f}")
check("a single pothole names its own repair method",
      single.summary.repair_method != "Multiple",
      f"got {single.summary.repair_method}")
check("cost is above zero", single.summary.repair_cost_idr > 0)
check("the mask travels with the measurement so the 3D view can use it",
      single.potholes[0].mask.shape == (IMAGE_SIZE, IMAGE_SIZE))

# ── Several potholes ──────────────────────────────────────────────────────────
several = build(n_potholes=3)

check("three potholes in, three measurements out", len(several.potholes) == 3)
check("area sums across potholes",
      abs(several.summary.area_cm2 - 300.0) < 0.1, f"got {several.summary.area_cm2:.4f}")
check("volume sums across potholes",
      abs(several.summary.volume_cm3 - 900.0) < 0.5,
      f"got {several.summary.volume_cm3:.4f}")
check("depth averages rather than sums",
      abs(several.summary.avg_depth_cm - 3.0) < 0.01,
      f"got {several.summary.avg_depth_cm:.4f}, summing would give 9")
check("cost sums across potholes",
      several.summary.repair_cost_idr > single.summary.repair_cost_idr)
check("several potholes report Multiple as the method",
      several.summary.repair_method == "Multiple",
      f"got {several.summary.repair_method}")
check("the summary severity is one of the four levels",
      several.summary.severity_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
      f"got {several.summary.severity_level}")

# ── Material classifier ───────────────────────────────────────────────────────
# Losing the classifier must degrade the result, not block it. An unweighted network
# would answer confidently and at random, so a missing one is passed as None instead.
without_material = build(material=None)
check("a missing material classifier falls back to asphalt",
      without_material.potholes[0].surface_type == "asphalt")
check("a missing material classifier reports zero confidence",
      without_material.potholes[0].surface_confidence == 0.0)

confident = build(material=StubMaterialClassifier("concrete", 0.91))
check("a confident material prediction is used",
      confident.potholes[0].surface_type == "concrete",
      f"got {confident.potholes[0].surface_type}")
check("a confident prediction keeps its confidence",
      abs(confident.potholes[0].surface_confidence - 0.91) < 1e-6)

unsure = build(material=StubMaterialClassifier("concrete", 0.40))
check("a prediction below the threshold falls back to asphalt",
      unsure.potholes[0].surface_type == "asphalt",
      f"got {unsure.potholes[0].surface_type}")
check("a rejected prediction still reports what it scored",
      abs(unsure.potholes[0].surface_confidence - 0.40) < 1e-6)

# ── Caller metadata ───────────────────────────────────────────────────────────
check("metadata the caller fills in later starts empty",
      single.image_name == "unknown"
      and single.annotated_path is None
      and single.latitude is None
      and single.db_id is None)
check("the result is the documented type", isinstance(single, AnalysisResult))

# ── Report ────────────────────────────────────────────────────────────────────
failed = [r for r in results if not r[1]]

for name, ok, detail in results:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail and not ok:
        line += f"  ({detail})"
    print(line)

print()
print(f"{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
