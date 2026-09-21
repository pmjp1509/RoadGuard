"""The analysis pipeline: one photo in, one measured result out.

Everything between a JPEG and a repair estimate lives here: segmentation, depth,
calibration, volume, severity, material, and cost. Callers construct a `Pipeline` once
and call `analyze` per image. Nothing in this module imports Streamlit, so the same code
serves the web app, a batch script, and the tests.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.core.calibration import Calibrator
from src.core.repair_advisor import RepairAdvisor, RepairRecommendation
from src.core.severity import SeverityClassifier, SeverityResult
from src.core.volumetric import VolumetricCalculator, VolumetricResult
from src.utils.logger import setup_logger

logger = setup_logger("Pipeline")

POTHOLE_CLASS = 0
REFERENCE_CLASS = 1
DEFAULT_SURFACE = "asphalt"


@dataclass
class PotholeResult:
    """One pothole, measured."""
    volumetric: VolumetricResult
    severity: SeverityResult
    repair: RepairRecommendation
    surface_type: str
    surface_confidence: float
    mask: np.ndarray


@dataclass
class AnalysisSummary:
    """One photo's totals.

    Area, volume, material, and cost add up across potholes. Depth does not, so it is
    the mean. Severity belongs to the worst hole in the photo, not to the photo.
    """
    area_cm2: float
    avg_depth_cm: float
    volume_cm3: float
    severity_level: str
    severity_score: float
    repair_method: str
    repair_cost_idr: float
    repair_material_kg: float


@dataclass
class AnalysisResult:
    """What `Pipeline.analyze` returns.

    `scaled` is the field to check first. When it is False no reference object was
    found, `potholes` is empty and `summary` is None: the images are still worth showing
    but there is no scale, so there are no centimetres to report.
    """
    annotated: np.ndarray
    depth_viz: np.ndarray
    depth_raw: np.ndarray
    original_rgb: np.ndarray
    pothole_count: int
    scaled: bool
    potholes: List[PotholeResult] = field(default_factory=list)
    summary: Optional[AnalysisSummary] = None

    # Filled in by the caller after analysis: where the image came from and went.
    image_name: str = "unknown"
    annotated_path: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    db_id: Optional[int] = None


class Pipeline:
    """Turns a photo into measurements.

    Construct it once, since loading the models is the expensive part, then call
    `analyze` per image.
    """

    def __init__(self, segmenter, depth_estimator, material_classifier, config: dict):
        """
        Args:
            segmenter: PotholeSegmenter, already loaded.
            depth_estimator: DepthEstimator, already loaded.
            material_classifier: MaterialClassifier, or None. When None the road surface
                is assumed to be asphalt, which changes only which repair material is
                recommended.
            config: parsed config.yaml.
        """
        self.segmenter = segmenter
        self.depth_estimator = depth_estimator
        self.material_classifier = material_classifier
        self.calculator = VolumetricCalculator(
            config["volumetric"]["calibration_constant"]
        )
        self.severity = SeverityClassifier()
        self.advisor = RepairAdvisor()
        self.material_threshold = (
            config.get("models", {}).get("material", {}).get("confidence_threshold", 0.6)
        )

    def analyze(self, image_rgb: np.ndarray) -> Optional[AnalysisResult]:
        """Measure every pothole in one RGB image.

        Returns None when the image holds no pothole. Returns a result with
        `scaled=False` when it holds potholes but no reference object.
        """
        detection = self.segmenter.detect(image_rgb, visualize=True)
        potholes = [d for d in detection["detections"] if d.class_id == POTHOLE_CLASS]
        if not potholes:
            return None

        depth_map = self.depth_estimator.predict(image_rgb)
        reference = self.segmenter.get_largest_detection(
            detection["detections"], REFERENCE_CLASS
        )

        base = dict(
            annotated=detection["annotated_image"],
            depth_viz=self.depth_estimator.visualize_depth(depth_map),
            depth_raw=depth_map,
            original_rgb=image_rgb,
            pothole_count=len(potholes),
        )

        # Every figure in cm traces back to the reference object's known real area.
        # Without one there is no scale, so stop before inventing numbers. An empty mask
        # counts as no reference: a detection too small to cover a pixel carries no area
        # to calibrate against, and dividing by it would raise out of the whole photo.
        if reference is None or np.sum(reference.mask) == 0:
            logger.info("No usable reference object detected; reporting images only.")
            return AnalysisResult(scaled=False, **base)

        reference_type = Calibrator.detect_reference_type(
            reference.mask, reference.bbox
        )
        # A neighbouring hole inside this pothole's bounding box would otherwise count
        # as flat road when the ground plane is estimated.
        all_potholes = np.zeros_like(potholes[0].mask)
        for pothole in potholes:
            all_potholes |= pothole.mask

        measured = [
            self._measure(
                pothole,
                reference.mask,
                reference_type,
                depth_map,
                image_rgb,
                others=(all_potholes & ~pothole.mask.astype(bool)).astype(np.uint8),
            )
            for pothole in potholes
        ]

        return AnalysisResult(
            scaled=True,
            potholes=measured,
            summary=self._summarize(measured),
            **base,
        )

    def _measure(self, pothole, reference_mask, reference_type, depth_map, image_rgb,
                 others=None):
        volumetric = self.calculator.calculate_volume(
            pothole.mask, reference_mask, pothole.bbox, depth_map, reference_type,
            exclude_mask=others,
        )

        surface_type, surface_confidence = self._classify_surface(
            pothole.bbox, image_rgb
        )

        severity = self.severity.classify(
            volumetric.avg_depth_cm, volumetric.area_cm2, volumetric.volume_cm3
        )
        repair = self.advisor.recommend(
            volumetric.volume_cm3,
            volumetric.avg_depth_cm,
            volumetric.area_cm2,
            severity_level=severity.level,
            surface_type=surface_type,
        )

        return PotholeResult(
            volumetric=volumetric,
            severity=severity,
            repair=repair,
            surface_type=surface_type,
            surface_confidence=surface_confidence,
            mask=pothole.mask,
        )

    def _classify_surface(self, bbox, image_rgb):
        """Classify the road surface from the pothole's own patch.

        Falls back to asphalt when the classifier is missing or unsure, since a wrong
        surface only picks a different repair material.
        """
        if self.material_classifier is None:
            return DEFAULT_SURFACE, 0.0

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2 = min(image_rgb.shape[1], int(x2))
        y2 = min(image_rgb.shape[0], int(y2))
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return DEFAULT_SURFACE, 0.0

        prediction = self.material_classifier.predict(crop)
        confidence = prediction["confidence"]
        if confidence >= self.material_threshold:
            return prediction["class"], confidence
        return DEFAULT_SURFACE, confidence

    @staticmethod
    def _summarize(potholes: List[PotholeResult]) -> AnalysisSummary:
        worst = max(potholes, key=lambda p: p.severity.score)
        return AnalysisSummary(
            area_cm2=sum(p.volumetric.area_cm2 for p in potholes),
            avg_depth_cm=sum(p.volumetric.avg_depth_cm for p in potholes) / len(potholes),
            volume_cm3=sum(p.volumetric.volume_cm3 for p in potholes),
            severity_level=worst.severity.level,
            severity_score=worst.severity.score,
            repair_method=worst.repair.method if len(potholes) == 1 else "Multiple",
            repair_cost_idr=sum(p.repair.total_cost_idr for p in potholes),
            repair_material_kg=sum(p.repair.material_kg for p in potholes),
        )
