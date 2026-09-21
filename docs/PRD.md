# InfraSight product requirements document

Version 1.1. Written 2026-09-06, based on commit `2c8f3fc` plus the fixes made the same
day for the reference object fallback, the depth axis direction, and depth precision.

This document describes the system as it currently behaves, together with the gaps that
remain open. Where a requirement is claimed but unverified, it says so.

## 1. Problem

Road maintenance agencies in Indonesia set pothole repair priority by eye. An inspector
looks at a hole, judges whether it is bad, and writes a note. Two things go wrong.

First, nobody measures anything. Without depth and volume there is no way to order
repair material or estimate cost before the crew arrives, so material gets over-ordered
or under-ordered and the crew comes back a second time.

Second, the record cannot be compared. "Deep pothole on Jl. Merdeka" cannot be ranked
against "wide pothole on Jl. Sudirman", so priority goes to whoever complained loudest.

Equipment that measures properly (LiDAR, stereo rigs, laser profilometers) costs more
than a district maintenance budget. A phone camera does not.

## 2. What InfraSight does

The inspector photographs a pothole with an everyday object next to it for scale, either
an ATM or KTP card or a Rp500 coin. InfraSight returns:

- surface area in cm², average depth in cm, volume in cm³
- a severity level (LOW, MEDIUM, HIGH, CRITICAL) with a score from 1 to 10
- a repair method, material quantity in kg, and cost in IDR
- a 3D view of the hole rebuilt from the depth map
- a PDF report, and a map pin when the photo carries GPS EXIF data

It runs as a Streamlit application, either locally on a machine with an NVIDIA GPU or as a
CPU-only container. The container costs about 2 s per photo, which is what makes a free
hosted demo practical. See `docs/DEPLOY.md`.

## 3. Goals and non-goals

### Goals

| ID | Goal | How it is measured |
|----|------|--------------------|
| G1 | Measure pothole volume from one photo, no special hardware | Volume error within ±30% of hand measurement |
| G2 | Rank potholes consistently | The same photo gives the same severity level on repeat runs |
| G3 | Give a crew a repair estimate they can act on | Method, kg, and IDR appear in every report |
| G4 | Keep a searchable record of past inspections | Every analysis is stored in SQLite and listed in History |
| G5 | Run on one consumer GPU | Works on 6 GB VRAM, RTX 3050 class |
| G6 | Be reachable without local setup | A hosted CPU demo answers in seconds and needs no install |

### Non-goals

- Video input or continuous road scanning. One photo at a time, batched but still per photo.
- Accounts, login, or per-user data. The hosted demo is a shared, anonymous instance.
- Cracks, rutting, or ravelling. Potholes only.
- Survey-grade accuracy. Monocular depth is relative, not metric. See section 8.
- Work order dispatch or integration with an agency asset management system.

## 4. Users

The field inspector walks a road segment, photographs damage on a phone, and uploads the
batch at the end of the day. They need the numbers and the PDF. They are not a computer
vision practitioner and will not touch the thresholds unless told to.

The maintenance planner takes no photos. They open History and the Damage Map to decide
which segment gets this month's budget. They need ranking, total cost, and location.

The reviewer or examiner judges whether the measurement is defensible. They need the
method, the assumptions, and the error bounds.

## 5. Use cases

### UC-01 Analyze a single pothole photo

Actor: field inspector.
Precondition: the photo shows at least one pothole with a card or coin lying flat nearby.

1. The inspector opens the Analyze page and uploads one JPG or PNG.
2. The inspector presses START BATCH ANALYSIS.
3. YOLOv8m-seg segments the potholes and the reference object.
4. Depth Anything V2 Small produces a relative depth map.
5. The system derives pixels per cm² from the reference object mask and its known real area.
6. The system estimates the ground plane, then the pothole bottom, then the depth between them.
7. The system computes area, average depth, max depth, and volume.
8. The system classifies the road surface material from the cropped pothole patch.
9. The system assigns a severity level and score, then a repair method, material kg, and cost.
10. The system writes the annotated image to disk and one row to SQLite.
11. The system displays the annotated image, the metrics, and the repair recommendation.

Postcondition: one history row exists and a PDF can be downloaded.

Exceptions:

- No pothole detected. The system warns, names the file, and suggests lowering the confidence threshold. No row is written.
- No reference object detected. The system shows the detection and the depth map, states that measurement needs a card or coin, and writes no row. See UC-11.
- Any other failure. The file is skipped with an error message and the rest of the batch continues.

### UC-02 Analyze several photos in one run

Actor: field inspector.
Precondition: a day's photos are on disk.

1. The inspector selects multiple files in the uploader.
2. The inspector presses START BATCH ANALYSIS.
3. The system processes files in order and updates a progress bar and a status line per file.
4. Each file that succeeds follows UC-01 steps 3 to 10.
5. The system reports how many of N files succeeded.
6. Results appear as one expander per file, with the first one open.

Postcondition: one history row per successful file.

Exception: if no file succeeded, the system says so and points at the confidence threshold.

### UC-03 Inspect a photo holding several potholes

Actor: field inspector.
Precondition: UC-01 or UC-02 ran on an image where YOLO found more than one pothole.

1. The inspector expands the result for that image.
2. The system shows one tab per pothole.
3. The inspector opens a tab and reads that hole's own area, depth, volume, severity, material, and cost.
4. Image-level totals (summed area, summed volume, summed cost) appear in the summary. The severity shown for the image belongs to its worst hole.

Only the first pothole's average depth reaches the history row. Per-pothole detail lives
in the session, not the database.

### UC-04 View the 3D reconstruction

Actor: field inspector, reviewer.
Precondition: a pothole tab is open.

1. The inspector switches on the "Open 3D Model" toggle for that pothole.
2. The system builds a Plotly surface mesh from the depth map inside the pothole mask.
3. The inspector rotates and zooms the mesh in the browser.

The toggle exists because each mesh is expensive to hold in memory. Meshes are built on
request rather than for every pothole in a batch.

### UC-05 Tune detection sensitivity

Actor: field inspector.
Precondition: a previous run missed a visible pothole, or boxed something that is not one.

1. The inspector lowers the confidence threshold (default 0.25) to catch more, or raises it to catch less.
2. The inspector adjusts the IoU threshold (default 0.45) when overlapping boxes merge wrongly.
3. The inspector re-runs the analysis.

Moving either slider reloads the model cache.

### UC-06 Generate a PDF report

Actor: field inspector.
Precondition: an analysis result is on screen.

1. The inspector presses "Download PDF Report" under the result.
2. The system renders a PDF holding the annotated image, area, average depth, volume, severity, repair method, material kg, cost, and coordinates when they exist.
3. The browser downloads the file.

Exception: if rendering fails, the system warns and the on-screen result stays intact.

### UC-07 Review past inspections

Actor: maintenance planner.
Precondition: at least one analysis has been saved.

1. The planner opens the History page.
2. The system lists every record with ID, date, image name, severity, volume, cost, and coordinates.
3. The planner selects a record to see the annotated image beside its full metrics and repair information.
4. The system shows totals: number of analyses, count of CRITICAL, count of HIGH, and summed estimated cost.

Exception: if the annotated image is missing from disk, the detail panel warns and shows
the metrics alone.

### UC-08 Delete a record

Actor: maintenance planner.
Precondition: a record is open in the History detail view.

1. The planner presses "Delete this Entry".
2. The system removes the row and reloads the page.

Deletion happens immediately without a confirmation step, and it leaves the annotated
image file on disk.

### UC-09 See damage on a map

Actor: maintenance planner.
Precondition: at least one stored photo carried GPS EXIF data.

1. The planner opens the Damage Map page.
2. The system places a Folium marker at each stored coordinate, coloured by severity: CRITICAL red, HIGH orange, MEDIUM blue, LOW green.
3. The planner clicks a marker to read the image name, severity, volume, and cost.

Exception: if no record has coordinates, the system explains that GPS EXIF is required
and shows a default map centred on Jakarta.

### UC-10 Identify the road surface material

Actor: the system, during UC-01.
Precondition: a pothole bounding box has been cropped.

1. MobileNetV3-Small classifies the crop as asphalt, concrete, or paving.
2. If confidence reaches 0.6 or above, the predicted class is used.
3. Otherwise the system falls back to asphalt.
4. The chosen material decides which repair material the advisor recommends.

Postcondition: the surface type and its confidence appear with the metrics.

### UC-11 Handle a photo with no reference object

Actor: field inspector.
Precondition: the photo shows a pothole, but no card or coin was detected.

1. The system segments the potholes and builds the depth map as usual.
2. The system stops before the calibration step.
3. The system shows the annotated image and the relative depth map, both labelled as unscaled.
4. The system explains that area, depth, volume, material, and cost all come from the reference object and asks for a re-shoot with a card or coin lying flat and fully visible.
5. Nothing is written to the database.

Every figure in cm traces back to the reference object's known real area. Producing
numbers without it means inventing a scale, and an invented scale is worse than a blank
field because it looks like a measurement.

### UC-12 Export the history as CSV

Actor: maintenance planner.
Precondition: at least one analysis has been saved.

1. The planner opens the History page.
2. The planner presses "Download History as CSV".
3. The browser downloads the same columns shown in the table, named by today's date.

## 6. Functional requirements

| ID | Requirement | Implemented in |
|----|-------------|----------------|
| FR-01 | Segment potholes and reference objects from an RGB image | `src/models/yolo_segmentation.py` |
| FR-02 | Produce a relative depth map for the image | `src/models/depth_estimation.py` |
| FR-03 | Tell a card from a coin by bounding box aspect ratio, above 1.3 means card | `src/core/calibration.py` |
| FR-04 | Convert pixel area to cm² using the reference object's known area | `src/core/calibration.py` |
| FR-05 | Estimate the ground plane from reference depth plus intact road inside the bbox | `src/core/volumetric.py` |
| FR-06 | Estimate the pothole bottom from the deepest 10% of masked pixels | `src/core/volumetric.py` |
| FR-07 | Convert normalized depth difference to cm with a calibration constant, default 30.0 | `src/core/volumetric.py` |
| FR-08 | Compute volume as area times average depth | `src/core/volumetric.py` |
| FR-09 | Score severity as 0.40 depth plus 0.30 area plus 0.30 volume, mapped to four levels | `src/core/severity.py` |
| FR-10 | Recommend one of three FHWA repair methods by severity, depth, and volume | `src/core/repair_advisor.py` |
| FR-11 | Convert volume to material kg using density and a compaction factor | `src/core/repair_advisor.py` |
| FR-12 | Estimate total cost as material cost plus a per-method labour rate | `src/core/repair_advisor.py` |
| FR-13 | Classify road surface material into three classes | `src/models/material_classifier.py` |
| FR-14 | Render a 3D mesh of the pothole on demand | `src/visualization/mesh_engine.py` |
| FR-15 | Read latitude and longitude from EXIF | `src/utils/gps_utils.py` |
| FR-16 | Store every analysis in SQLite and support read and delete | `src/core/history_manager.py` |
| FR-17 | Render a PDF report | `src/core/report_generator.py` |
| FR-18 | Serve the Analyze, History, and Damage Map pages | `webapp/app.py` |
| FR-19 | Process an uploaded batch, isolating per-file failures | `webapp/app.py` |
| FR-20 | Refuse to report cm figures when no reference object is detected | `webapp/app.py` |
| FR-21 | Export the history table as CSV | `webapp/app.py` |
| FR-22 | Read the depth axis in the direction the model actually uses | `src/models/depth_estimation.py`, `scripts/probe_depth_convention.py` |
| FR-23 | Resolve weights locally, falling back to a Hugging Face model repo | `src/utils/weights.py` |
| FR-24 | Cap a batch so one visitor cannot exhaust memory | `webapp/app.py`, `app.max_batch_files` |
| FR-25 | Warn when the deployment's storage does not survive a restart | `webapp/app.py`, `INFRASIGHT_EPHEMERAL` |
| FR-26 | Run the whole analysis behind one call, independent of the interface | `src/pipeline.py` |

## 7. Non-functional requirements

| ID | Requirement | Status |
|----|-------------|--------|
| NFR-01 | Volume within ±30% of hand measurement | Claimed, not measured. No ground truth set exists. See L2. |
| NFR-02 | YOLO mAP@50 above 0.60 | Met by the fine-tuned weights, per the training runs |
| NFR-03 | Material classifier above 95% accuracy | Reported from training, not re-validated on field photos |
| NFR-04 | Runs on 6 GB VRAM | Met. 3D meshes are opt-in to keep memory down. |
| NFR-05 | First model load around 30 s, cached afterwards | Met through `st.cache_resource` |
| NFR-06 | Upload limit 10 MB per image | Set in config |
| NFR-07 | Python 3.10 or 3.11, PyTorch 2.6+, CUDA 12.4+ | Documented in the README |
| NFR-08 | English throughout the interface | Met |
| NFR-09 | Runs without a GPU | Met. Measured at about 2 s per photo and 10 s cold load on a desktop CPU. Input size barely matters, since both models resize internally. |
| NFR-10 | Anyone can reproduce the environment | Met. Pinned `requirements.txt` and a CPU-only Dockerfile, verified by building it: 2.92 GB, container healthy about 20 s after start. `fpdf2`, `pandas`, and `sqlalchemy` were previously imported but never declared. |
| NFR-11 | Weights are fetched, not committed | Met. `src/utils/weights.py` prefers a local file and falls back to a Hugging Face model repo. |

## 8. Known limitations

These describe how the current build behaves. They belong in any honest evaluation of
the system.

**L1. Closed.** A photo without a reference object used to fall back to the pothole's own
mask while keeping the card's real area, so it reported exactly 45.9 cm² every time. The
system now stops before calibration and says why. See UC-11 and FR-20.

**L2. The calibration constant is a guess.** The default of 30.0 was never fitted to
measured potholes. `set_calibration_constant()` exists and documents the fitting
procedure, but nothing in the repository performs it. Every depth and volume figure
inherits this uncertainty, so NFR-01 remains unproven.

**L3. Closed.** `_estimate_pothole_bottom` used to read the highest depth values as the
deepest pixels while `max_depth_cm` read the lowest, so the two disagreed.
`scripts/probe_depth_convention.py` compared the depth inside each pothole mask against a
ring of road around it on 12 Indonesian photos. Eleven put the hole below the road, which
confirms inverse depth: smaller values are farther away. Both estimates now read the
smallest values as deepest, and `tests/test_volumetric_calibration.py` fails if that flips
back.

Fixing this also uncovered a precision problem. The depth pipeline returned an 8-bit PIL
image, quantising the whole scene into 256 levels, and three of the twelve probe samples
came back with a hole-to-road difference of exactly zero. The estimator now takes the raw
float tensor, and no sample reads zero.

The renderer read the same axis and was missed on the first pass. `mesh_engine.py`
negated the ground-relative depth, so once the calculator flipped, the 3D view extruded
every pothole upward: the metric panel reported a depth while the mesh beside it showed a
mound. Both now share one direction, and the test asserts the surface sinks. Two unused
mesh builders carrying the same stale assumption were deleted rather than corrected.

**L4. Depth is relative, not metric.** Monocular estimation carries no absolute scale.
One scalar constant is applied regardless of camera height, tilt, or lens, so accuracy
shifts with how the photo was taken.

**L5. The reference object must lie flat and be fully visible.** A card at an angle,
partly shadowed, or partly buried gives a wrong pixel area, and that error propagates
into area, volume, kg, and cost.

**L6. Card versus coin detection uses aspect ratio alone.** A card shot at a steep angle
can project to a near-square shape and read as a coin, which swaps the reference area
from 45.9 cm² to 5.73 cm², an eightfold error.

**L7. Closed.** `avg_depth_cm` in the database used to come from the first pothole while
area, volume, and cost were image totals, so a multi-pothole row mixed two scopes. The
PDF and the database row now read the same `AnalysisSummary`, where depth is the mean
across potholes and the rest are sums. Per-pothole detail still lives only in the
session, not the database.

**L8. GPS depends on EXIF.** Screenshots, edited images, and photos taken with location
services off carry no coordinates and never reach the map. There is no manual coordinate
entry.

**L9. Costs are static.** Material prices and labour rates are hard-coded for the 2024
to 2025 market and cannot be changed from the interface.

**L10. Closed for the modules that produce numbers.** `tests/test_pipeline.py` drives the
whole analysis with stub models, 29 checks, no GPU and no weights.
`tests/test_volumetric_calibration.py` covers area conversion, the depth axis direction in
both the calculator and the renderer, unit conversion, and the empty-mask errors, 26 checks.
`tests/test_severity_repair.py` covers severity levels and repair costs. Nothing runs these
automatically; there is no CI.

**L11. The demo deployment forgets everything on restart.** Hugging Face Spaces resets the
container filesystem, which holds both the SQLite history and the annotated images.
Persistent storage there is a paid add-on. The app says so in the sidebar when
`INFRASIGHT_EPHEMERAL` is set, and the PDF and CSV downloads are how a result leaves the
demo. A self-hosted deployment with a mounted volume keeps its history normally.

**L12. One process, no queue.** Concurrent visitors share a single Streamlit process and
one copy of each model. A batch is capped at 10 photos so nobody can exhaust memory
alone, but simultaneous traffic just gets slower. The pipeline no longer depends on the
UI, so putting it behind a queue or an HTTP endpoint is now a matter of writing that
front end; this version does not include one.

## 9. Data model

SQLite at `data/infrasight.db`, one table.

`analysis_history`

| Column | Type | Notes |
|--------|------|-------|
| `id` | Integer | Primary key |
| `timestamp` | DateTime | Defaults to insert time |
| `image_name` | String(255) | Original upload filename |
| `image_path` | String(511) | Path to the saved annotated image |
| `area_cm2` | Float | Sum across all potholes in the image |
| `avg_depth_cm` | Float | First pothole only, see L7 |
| `volume_cm3` | Float | Sum across all potholes |
| `severity_level` | String(50) | LOW, MEDIUM, HIGH, or CRITICAL, from the worst hole |
| `severity_score` | Float | 1 to 10, from the worst hole |
| `repair_method` | String(100) | `"Multiple"` when the image holds more than one pothole |
| `repair_cost_idr` | Float | Sum across all potholes |
| `repair_material_kg` | Float | Sum across all potholes |
| `latitude` | Float | Nullable |
| `longitude` | Float | Nullable |
| `metadata_json` | Text | Nullable, reserved, currently unused |

Annotated images go to `data/processed/analysis_results/` and stay there when their
database row is deleted. Photos analysed without a reference object are never written to
the table, so no unscaled figure can reach the History page, the map, or a CSV export.

## 10. Out of scope for this version

- Video input and road segment scanning
- Manual entry of GPS coordinates
- Editing prices, labour rates, or severity thresholds from the interface
- Accounts, roles, or per-user history
- Automatic per-photo calibration from camera intrinsics

## 11. Open questions

Only one is left. Three others were settled on 2026-09-06: the depth axis was measured
rather than argued about, unscaled photos now refuse to report cm figures, and CSV export
was added because it cost three lines.

1. **How is the calibration constant fitted?** No ground truth exists yet. The three
   pothole datasets in `data/raw/` are YOLO format, a class index and box coordinates
   with no depth or volume field, so none of them can fit the constant. Whether any
   public dataset carries measured pothole volume has not been checked, and that search
   is worth doing before spending a day photographing.

   The fallback that does not depend on the search: a container of known dimensions
   placed on the road beside a reference card, photographed from several distances and
   angles. Its volume is known by construction rather than measured, so five containers
   of different sizes give a target to regress against with no measurement error of their
   own. This is waiting on photographs.

## 12. Success criteria

The system is done for its intended use when all of the following hold.

1. Volume error is measured against known-volume targets and stated as a number rather
   than assumed. Closes L2 and NFR-01. **Open**, waiting on photographs.
2. A photo without a reference object refuses to report cm figures. Closes L1. **Done.**
3. The depth convention is settled by experiment and both depth estimates agree. Closes
   L3. **Done**, 11 of 12 samples.
4. `volumetric.py` and `calibration.py` have tests that fail when the maths breaks.
   **Done**, 23 checks.
5. An inspector can go from photo to PDF without reading the source code. **Done.**
