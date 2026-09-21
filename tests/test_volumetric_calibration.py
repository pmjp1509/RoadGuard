"""Verification test for the volumetric and calibration modules.

These two modules produce every number the rest of the system reports, so the checks
here pin down the arithmetic and, more importantly, the direction of the depth axis.
Depth Anything V2 emits inverse depth, where smaller values sit farther from the camera.
Reading it the other way round makes a deep pothole look shallow.

Run: python tests/test_volumetric_calibration.py
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.calibration import Calibrator
from src.core.volumetric import VolumetricCalculator

ROAD_DEPTH = 0.8  # inverse depth of flat road: larger means nearer the camera

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))


def flat_scene():
    """459 reference pixels over 45.9 cm² gives exactly 10 pixels per cm².

    The pothole covers 1000 pixels, so 100 cm². Its floor sits 0.3 below the road in
    normalized depth, which a calibration constant of 10.0 turns into 3 cm.
    """
    depth = np.full((200, 200), ROAD_DEPTH, dtype=np.float32)

    ref = np.zeros((200, 200), dtype=np.uint8)
    ref[10:27, 10:37] = 1  # 17 x 27 = 459 px

    hole = np.zeros((200, 200), dtype=np.uint8)
    hole[100:140, 100:125] = 1  # 40 x 25 = 1000 px
    depth[hole == 1] = ROAD_DEPTH - 0.3

    return depth, ref, hole, (100, 100, 125, 140)


def graded_scene():
    """Same geometry, but the floor slopes from 0.7 down to 0.4 across the hole.

    The deepest tenth of the pixels sits near 0.41, so the reported depth should land
    near (0.8 - 0.41) * 10 = 3.9 cm. Reading the axis backwards would pick the shallow
    end near 0.69 instead and report roughly 1.1 cm.
    """
    depth = np.full((200, 200), ROAD_DEPTH, dtype=np.float32)

    ref = np.zeros((200, 200), dtype=np.uint8)
    ref[10:27, 10:37] = 1

    hole = np.zeros((200, 200), dtype=np.uint8)
    hole[100:120, 100:150] = 1  # 20 x 50 = 1000 px

    gradient = np.linspace(0.4, 0.7, 50, dtype=np.float32)
    depth[100:120, 100:150] = np.tile(gradient, (20, 1))

    return depth, ref, hole, (100, 100, 150, 120)


# ── Volumetric ────────────────────────────────────────────────────────────────
calc = VolumetricCalculator(calibration_constant=10.0)

depth, ref, hole, bbox = flat_scene()
flat = calc.calculate_volume(hole, ref, bbox, depth, "card")

check("flat: area is 100 cm²", abs(flat.area_cm2 - 100.0) < 0.01, f"got {flat.area_cm2:.4f}")
check("flat: depth is 3 cm", abs(flat.avg_depth_cm - 3.0) < 0.01, f"got {flat.avg_depth_cm:.4f}")
check("flat: volume is 300 cm³", abs(flat.volume_cm3 - 300.0) < 0.1, f"got {flat.volume_cm3:.4f}")
check("flat: volume equals area times depth",
      abs(flat.volume_cm3 - flat.area_cm2 * flat.avg_depth_cm) < 1e-6)
check("flat: card reference reports high confidence", flat.confidence == "High")

depth, ref, hole, bbox = graded_scene()
graded = calc.calculate_volume(hole, ref, bbox, depth, "card")

check("graded: depth follows the deep end, not the shallow end",
      graded.avg_depth_cm > 3.0,
      f"got {graded.avg_depth_cm:.4f} cm, expected above 3.0 (backwards axis gives ~1.1)")
check("graded: max depth reaches the deepest pixel",
      abs(graded.max_depth_cm - 4.0) < 0.01, f"got {graded.max_depth_cm:.4f}")
check("graded: max depth is at least average depth",
      graded.max_depth_cm >= graded.avg_depth_cm - 1e-6)

coin = calc.calculate_volume(hole, ref, bbox, depth, "coin_rp500")
check("coin reference reports low confidence", coin.confidence == "Low")
check("a smaller reference object makes the same pothole measure smaller",
      coin.area_cm2 < graded.area_cm2,
      f"coin {coin.area_cm2:.2f} vs card {graded.area_cm2:.2f}")

empty = np.zeros((200, 200), dtype=np.uint8)
try:
    calc.calculate_volume(empty, ref, bbox, depth, "card")
    check("empty pothole mask is rejected", False, "no error raised")
except ValueError:
    check("empty pothole mask is rejected", True)

try:
    calc.calculate_volume(hole, empty, bbox, depth, "card")
    check("empty reference mask is rejected", False, "no error raised")
except ValueError:
    check("empty reference mask is rejected", True)

# ── 3D mesh ───────────────────────────────────────────────────────────────────
# The renderer reads the same depth axis as the calculator. When only one of the two
# was flipped, the metric panel reported a depth while the mesh beside it showed a mound.
from src.visualization.mesh_engine import Mesh3DVisualizer

bowl_depth = np.full((200, 200), ROAD_DEPTH, dtype=np.float32)
bowl_mask = np.zeros((200, 200), dtype=np.uint8)
yy, xx = np.mgrid[0:200, 0:200]
radius = np.sqrt(((yy - 100) / 40) ** 2 + ((xx - 100) / 40) ** 2)
bowl_mask[radius < 1.0] = 1
bowl_depth[radius < 1.0] -= 0.3 * (1.0 - radius[radius < 1.0])

figure = Mesh3DVisualizer.create_premium_pothole_mesh(
    bowl_depth, bowl_mask,
    metrics={"depth": 5.0, "area": 300.0, "severity": "HIGH"},
)
mesh_z = np.array(figure.data[0].z, dtype=float)

check("mesh: the pothole sinks below the road instead of bulging out of it",
      np.nanmin(mesh_z) < -0.01,
      f"lowest point {np.nanmin(mesh_z):.4f}, expected clearly negative")
check("mesh: the road around it stays at zero",
      abs(np.nanmax(mesh_z)) < 0.01, f"highest point {np.nanmax(mesh_z):.4f}")

try:
    Mesh3DVisualizer.create_premium_pothole_mesh(
        bowl_depth, np.zeros((200, 200), dtype=np.uint8)
    )
    check("mesh: an empty mask is rejected", False, "no error raised")
except ValueError:
    check("mesh: an empty mask is rejected", True)

# ── Calibration ───────────────────────────────────────────────────────────────
card_mask = np.zeros((200, 200), dtype=np.uint8)
card_mask[50:84, 50:103] = 1  # 34 x 53, aspect ratio about 1.56

coin_mask = np.zeros((200, 200), dtype=np.uint8)
coin_mask[50:80, 50:80] = 1  # square, aspect ratio 1.0

check("a card-shaped mask reads as a card",
      Calibrator.detect_reference_type(card_mask) == "card")
check("a square mask reads as a coin",
      Calibrator.detect_reference_type(coin_mask) == "coin_rp500")
check("a card-shaped bbox reads as a card",
      Calibrator.detect_reference_type(card_mask, (50, 50, 103, 84)) == "card")

check("card area is 45.9 cm²", Calibrator.get_reference_specs("card")["area_cm2"] == 45.9)
check("coin area is 5.73 cm²", Calibrator.get_reference_specs("coin_rp500")["area_cm2"] == 5.73)

try:
    Calibrator.get_reference_specs("brick")
    check("an unknown reference type is rejected", False, "no error raised")
except ValueError:
    check("an unknown reference type is rejected", True)

px_per_cm2 = Calibrator.calculate_pixels_per_cm2(ref, "card")
check("459 reference pixels give 10 pixels per cm²",
      abs(px_per_cm2 - 10.0) < 1e-6, f"got {px_per_cm2:.6f}")
check("2000 pixels convert back to 200 cm²",
      abs(Calibrator.pixels_to_cm2(2000, px_per_cm2) - 200.0) < 1e-6)

try:
    Calibrator.calculate_pixels_per_cm2(empty, "card")
    check("an empty reference mask is rejected", False, "no error raised")
except ValueError:
    check("an empty reference mask is rejected", True)

check("a card reference is high confidence", Calibrator.get_confidence_level("card") == "High")
check("a coin reference is low confidence",
      Calibrator.get_confidence_level("coin_rp500") == "Low")

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
