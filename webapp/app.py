"""
InfraSight — Pothole Volumetric Analysis
Premium Dashboard: Multi-page Navigation, Batch Processing, History & Map.
"""
import os
import sys
import yaml
import tempfile
import cv2
import torch
import numpy as np
import pandas as pd
import folium
import streamlit as st

from PIL import Image
from pathlib import Path
from datetime import datetime
from streamlit_folium import st_folium

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.absolute()
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline                   import Pipeline, AnalysisResult
from src.models.yolo_segmentation  import PotholeSegmenter
from src.models.depth_estimation    import DepthEstimator
from src.models.material_classifier import MaterialClassifier
from src.core.history_manager       import HistoryManager
from src.core.report_generator      import ReportGenerator
from src.visualization.mesh_engine  import Mesh3DVisualizer
from src.utils.gps_utils            import extract_gps
from src.utils.logger               import setup_logger
from src.utils.weights              import resolve_from_config as resolve_weights

logger = setup_logger("WebApp")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfraSight Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top right, #1e293b, #0f172a);
    color: #f8fafc;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
}

.metric-title {
    color: #94a3b8;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 4px;
}

.severity-badge {
    padding: 6px 14px;
    border-radius: 99px;
    font-weight: 700;
    font-size: 0.8rem;
    display: inline-block;
    letter-spacing: 0.04em;
}

.premium-header {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(to right, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 700;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px -10px #3b82f6;
    opacity: 0.9;
}

[data-testid="stMetricValue"] { font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ── Singletons ─────────────────────────────────────────────────────────────────
history_mgr = HistoryManager()
report_gen  = ReportGenerator()


# ── Config ─────────────────────────────────────────────────────────────────────
@st.cache_data
def _app_config() -> dict:
    """The `app` block of config.yaml, readable without loading any model.

    A deployment sets INFRASIGHT_EPHEMERAL=1 rather than editing the committed file,
    so the same commit runs locally without claiming its storage disappears.
    """
    app_cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            app_cfg = (yaml.safe_load(f) or {}).get("app", {})

    if os.environ.get("INFRASIGHT_EPHEMERAL", "").lower() in ("1", "true", "yes"):
        app_cfg["ephemeral_storage"] = True

    return app_cfg


# ── Pipeline loader (cached) ───────────────────────────────────────────────────
# max_entries caps how many model sets can be alive at once. Each entry holds YOLO,
# Depth Anything and MobileNetV3, around 700 MB, and the cache is keyed on the two
# sliders: without a cap, nudging a slider a few times exhausts a small host.
@st.cache_resource(max_entries=2)
def get_pipeline(conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    """Load the models once and hand back the analysis pipeline.

    Raises rather than returning None on failure. Returning None would let
    `cache_resource` store the failure: one network blip while fetching weights would
    then keep the app dead for that threshold pair until the process restarts.
    """
    with st.spinner("Initializing AI models…"):
        if not CONFIG_PATH.exists():
            raise RuntimeError(f"Config not found: {CONFIG_PATH}")

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        yolo_cfg = config["models"]["yolo"]
        weights = resolve_weights(
            yolo_cfg, ("weights_path", "weights_fallback", "weights_legacy")
        )

        logger.info(f"Loading YOLO from {Path(weights).name}...")
        segmenter = PotholeSegmenter(
            weights, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )

        depth_cfg = config["models"]["depth"]
        device = depth_cfg.get("device", "auto")
        if device == "auto":
            device = None  # DepthEstimator picks cuda when it is available
        elif device == "cuda" and not torch.cuda.is_available():
            # A config carrying "cuda" on a CPU-only host used to be ignored entirely.
            # Now that the field is read, honouring it literally would break the load.
            logger.warning("config asks for cuda but no GPU is visible; using CPU.")
            device = "cpu"

        logger.info("Loading Depth Anything V2 (first run may take ~30 s)...")
        depth_est = DepthEstimator(depth_cfg["model_name"], device=device)

        # Material classification only steers which repair material is recommended, so
        # losing it degrades the result rather than invalidating it. Running an
        # unweighted network instead would produce confident-looking noise.
        logger.info("Loading Material Classifier...")
        try:
            mat_clf = MaterialClassifier(
                model_path=resolve_weights(config["models"]["material"])
            )
        except FileNotFoundError as exc:
            logger.warning(f"Material classifier unavailable: {exc}")
            st.warning(
                "Material classifier weights unavailable, so road surface is assumed to "
                f"be asphalt. Everything else is unaffected. ({exc})"
            )
            mat_clf = None

        return Pipeline(segmenter, depth_est, mat_clf, config)


# ── Helper: report and history payloads ────────────────────────────────────────
def _report_fields(res: AnalysisResult) -> dict:
    """The fields the PDF and the history row share, from one place."""
    return {
        "area_cm2":           res.summary.area_cm2,
        "avg_depth_cm":       res.summary.avg_depth_cm,
        "volume_cm3":         res.summary.volume_cm3,
        "severity_level":     res.summary.severity_level,
        "severity_score":     res.summary.severity_score,
        "repair_method":      res.summary.repair_method,
        "repair_cost_idr":    res.summary.repair_cost_idr,
        "repair_material_kg": res.summary.repair_material_kg,
        "latitude":           res.latitude,
        "longitude":          res.longitude,
    }


def _make_pdf_bytes(res: AnalysisResult) -> bytes:
    report_data = {
        "image_name":     res.image_name,
        "annotated_path": res.annotated_path,
        **_report_fields(res),
    }
    pdf_path = report_gen.generate_pdf_report(report_data)
    with open(pdf_path, "rb") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Analyze
# ══════════════════════════════════════════════════════════════════════════════
def page_analyze():
    st.markdown('<h1 class="premium-header">Analyze</h1>', unsafe_allow_html=True)
    st.markdown("Upload one or more photos for pothole tomography & volumetric analysis.")

    # ── Threshold controls ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        conf_t = st.slider(
            "Confidence Threshold", 0.05, 1.0, 0.25, 0.05,
            help="Lower = more detections but more noise"
        )
    with col2:
        iou_t = st.slider(
            "IoU Threshold", 0.1, 1.0, 0.45, 0.05,
            help="Intersection-over-Union for overlapping bounding boxes"
        )

    try:
        pipeline = get_pipeline(conf_t, iou_t)
    except Exception as exc:
        # Nothing is cached on failure, so a retry after the cause is fixed works.
        st.error(f"Could not load the models: {exc}")
        st.stop()

    # ── File uploader ─────────────────────────────────────────────────
    files = st.file_uploader(
        "Upload Images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    max_files = _app_config().get("max_batch_files", 10)
    if files and len(files) > max_files:
        st.error(
            f"{len(files)} files selected, but this instance accepts {max_files} at a "
            "time. Each image loads a full depth map into memory. Upload them in "
            "smaller batches, or raise app.max_batch_files in config/config.yaml on a "
            "machine with more RAM."
        )
        files = None

    if files and st.button("START BATCH ANALYSIS"):
        progress = st.progress(0)
        status   = st.empty()
        batch_results = []

        for i, file in enumerate(files):
            status.text(f"Processing {file.name} ({i + 1}/{len(files)})…")
            img_np = np.array(Image.open(file).convert("RGB"))

            try:
                res = pipeline.analyze(img_np)
                if res is None:
                    st.warning(f"{file.name}: No potholes detected (conf > {conf_t})")
                    progress.progress((i + 1) / len(files))
                    continue

                res.image_name = file.name

                if not res.scaled:
                    batch_results.append(res)
                    st.warning(
                        f"{file.name}: found {res.pothole_count} pothole(s) but no "
                        "reference object. Measurements need a card or coin lying flat "
                        "beside the pothole for scale, so no size figures were produced."
                    )
                    progress.progress((i + 1) / len(files))
                    continue

                # Save annotated image
                save_dir = ROOT_DIR / "data" / "processed" / "analysis_results"
                save_dir.mkdir(parents=True, exist_ok=True)
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = save_dir / f"annotated_{ts}_{file.name}"
                cv2.imwrite(
                    str(img_path),
                    cv2.cvtColor(res.annotated, cv2.COLOR_RGB2BGR)
                )
                res.annotated_path = str(img_path)

                # GPS extraction. extract_gps reads from a path, so the upload has to
                # touch disk. A temp file keeps two same-named uploads apart and is
                # cleaned up even when extraction raises.
                suffix = Path(file.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(file.getbuffer())
                    tmp_path = Path(tmp.name)
                try:
                    res.latitude, res.longitude = extract_gps(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)

                res.db_id = history_mgr.save_analysis({
                    "image_name": file.name,
                    "image_path": str(img_path),
                    **_report_fields(res),
                })
                batch_results.append(res)

            except Exception as exc:
                st.error(f"{file.name}: Error — {exc}")

            progress.progress((i + 1) / len(files))

        if batch_results:
            measured = sum(1 for r in batch_results if r.scaled)
            summary = f"{measured}/{len(files)} images measured."
            if measured < len(batch_results):
                summary += (
                    f" {len(batch_results) - measured} had potholes but no reference "
                    "object, so they were not measured or saved."
                )
            status.success(summary)
            st.session_state["batch_results"] = batch_results
        else:
            status.error(
                "No images were successfully processed. "
                "Make sure the images clearly show potholes, "
                "or lower the Confidence Threshold."
            )

    # ── Display results ───────────────────────────────────────────────
    if "batch_results" not in st.session_state:
        return

    for i, res in enumerate(st.session_state["batch_results"]):
        if not res.scaled:
            label = (
                f"📸 {res.image_name}  —  "
                f"{res.pothole_count} pothole(s) detected  —  "
                f"No reference object, not measured"
            )
            with st.expander(label, expanded=(i == 0)):
                st.image(res.annotated, caption="Detection Result (YOLO)",
                         use_container_width=True)
                st.image(res.depth_viz,
                         caption="Relative depth map (no scale, cannot be read in cm)",
                         use_container_width=True)
                st.warning(
                    "Area, depth, volume, repair material and cost all come from the "
                    "reference object, so none of them can be computed for this photo. "
                    "Take it again with an ATM/KTP card or a Rp500 coin lying flat next "
                    "to the pothole, fully visible."
                )
            continue

        n_holes = len(res.potholes)
        label   = (
            f"📸 {res.image_name}  —  "
            f"{n_holes} pothole(s) detected  —  "
            f"Severity: {res.summary.severity_level}"
        )
        with st.expander(label, expanded=(i == 0)):
            # Top row: annotated image
            st.image(res.annotated, caption="Detection Result (YOLO)", use_container_width=True)

            st.markdown("---")

            # Per-pothole tabs
            tabs = st.tabs([f"Pothole #{p + 1}" for p in range(n_holes)])
            for p_idx, (tab, p_res) in enumerate(zip(tabs, res.potholes)):
                with tab:
                    col_3d, col_info = st.columns([1, 1])

                    # ── 3-D visualisation ────────────────────────────
                    with col_3d:
                        st.markdown("#### 🔬 3D Tomography")
                        show_3d = st.toggle("Open 3D Model", key=f"toggle_3d_{i}_{p_idx}")
                        
                        if show_3d:
                            viz = Mesh3DVisualizer()
                            fig = viz.create_premium_pothole_mesh(
                                res.depth_raw,
                                p_res.mask,
                                metrics={
                                    "depth":    p_res.volumetric.avg_depth_cm,
                                    "area":     p_res.volumetric.area_cm2,
                                    "severity": p_res.severity.level,
                                }
                            )
                            st.plotly_chart(
                                fig, use_container_width=True,
                                key=f"plotly_{i}_{p_idx}"
                            )
                        else:
                            st.info("3D Tomography is disabled to save memory. Click the toggle above to view.")

                    # ── Metrics & repair info ─────────────────────────
                    with col_info:
                        sev = p_res.severity
                        vol = p_res.volumetric
                        st.markdown("#### 📊 Metrics & Recommendations")
                        st.markdown(
                            f'<span class="severity-badge" '
                            f'style="background:{sev.color};color:white">'
                            f'SEVERITY: {sev.level}</span>',
                            unsafe_allow_html=True
                        )
                        st.write("")

                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Severity Score", f"{sev.score}/10")
                        mc2.metric("Avg. Depth", f"{vol.avg_depth_cm:.1f} cm")
                        mc3.metric("Volume", f"{vol.volume_cm3:.0f} cm³")

                        st.metric("Surface Area", f"{vol.area_cm2:.1f} cm²")

                        st.markdown("---")
                        rep = p_res.repair
                        st.write("**🔧 Repair Method:**", rep.method)
                        st.write(
                            f"**Road Material:** {p_res.surface_type.capitalize()} "
                            f"(conf {p_res.surface_confidence:.2f})"
                        )
                        st.metric("Estimated Cost", f"Rp {rep.total_cost_idr:,.0f}")

                        st.markdown("**Material Details:**")
                        st.write(f"- {rep.material_name}: **{rep.material_kg:.2f} kg**")
                        st.write(
                            f"- Sealant / Tack Coat: "
                            f"**{vol.area_cm2 * 0.0001:.3f} L**"
                        )
                        st.write(f"- Est. time: **{rep.estimated_time_hours:.1f} hours**")
                        st.write(f"- Durability: **{rep.durability_months} months**")

                        if sev.notes if hasattr(sev, "notes") else False:
                            st.info(sev.notes)

            # ── PDF download ─────────────────────────────────────────
            # Streamlit runs an expander's body on every rerun whether or not it is
            # open, and each call writes a fresh timestamped PDF to reports/. Building
            # it behind a button keeps a slider nudge from re-rendering the whole batch.
            st.markdown("---")
            if not st.session_state.get(f"pdf_ready_{i}"):
                st.button(
                    f"Prepare PDF Report — {res.image_name}",
                    key=f"pdf_prep_{i}",
                    on_click=lambda idx=i: st.session_state.__setitem__(
                        f"pdf_ready_{idx}", True
                    ),
                )
            else:
                try:
                    st.download_button(
                        label=f"⬇️ Download PDF Report — {res.image_name}",
                        data=_make_pdf_bytes(res),
                        file_name=f"Report_{res.image_name}.pdf",
                        mime="application/pdf",
                        key=f"dl_btn_{i}",
                    )
                except Exception as exc:
                    st.warning(f"Failed to generate PDF: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: History
# ══════════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown('<h1 class="premium-header">History</h1>', unsafe_allow_html=True)

    history = history_mgr.get_all_history()
    if not history:
        st.info("No analysis history available yet.")
        return

    df_rows = []
    for h in history:
        df_rows.append({
            "ID":          h.id,
            "Date":        h.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Image":       h.image_name,
            "Severity":    h.severity_level,
            "Volume (cm³)":f"{h.volume_cm3:.1f}",
            "Cost (IDR)":  f"{h.repair_cost_idr:,.0f}",
            "Lat":         f"{h.latitude:.5f}" if h.latitude else "—",
            "Lon":         f"{h.longitude:.5f}" if h.longitude else "—",
        })

    df = pd.DataFrame(df_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download History as CSV",
        df.to_csv(index=False).encode("utf-8"),
        f"infrasight_history_{datetime.now():%Y%m%d}.csv",
        "text/csv",
    )

    # ── Detailed View Section ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Search & Detailed View")
    
    if not df.empty:
        # Create option list for selectbox: "ID - ImageName"
        options = [f"{h.id} - {h.image_name}" for h in history]
        selected_option = st.selectbox("Select an entry to view details:", options)
        
        if selected_option:
            selected_id = int(selected_option.split(" - ")[0])
            # Find the history object
            h_detail = next((h for h in history if h.id == selected_id), None)
            
            if h_detail:
                col_img, col_metrics = st.columns([1, 1])
                
                with col_img:
                    if h_detail.image_path and Path(h_detail.image_path).exists():
                        st.image(h_detail.image_path, caption=f"Analyzed Image: {h_detail.image_name}", use_container_width=True)
                    else:
                        st.warning("Annotated image file not found on disk.")
                
                with col_metrics:
                    st.markdown(f"#### Analysis Details (ID: {h_detail.id})")
                    
                    # Severity Badge
                    color_map = {"CRITICAL": "#9C27B0", "HIGH": "#F44336", "MEDIUM": "#FF9800", "LOW": "#4CAF50"}
                    color = color_map.get(h_detail.severity_level, "#94a3b8")
                    st.markdown(
                        f'<span class="severity-badge" style="background:{color};color:white">'
                        f'SEVERITY: {h_detail.severity_level}</span>',
                        unsafe_allow_html=True
                    )
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Area", f"{h_detail.area_cm2:.1f} cm²")
                    m1.metric("Avg. Depth", f"{h_detail.avg_depth_cm:.1f} cm")
                    m2.metric("Volume", f"{h_detail.volume_cm3:.1f} cm³")
                    m2.metric("Score", f"{h_detail.severity_score:.1f}/10")
                    
                    st.markdown("---")
                    st.markdown("**🔧 Maintenance Info**")
                    st.write(f"**Method:** {h_detail.repair_method}")
                    st.write(f"**Est. Cost:** Rp {h_detail.repair_cost_idr:,.0f}")
                    st.write(f"**Material Needed:** {h_detail.repair_material_kg:.2f} kg")
                    
                    st.markdown("---")
                    st.markdown("**📍 Location**")
                    loc_str = f"{h_detail.latitude:.6f}, {h_detail.longitude:.6f}" if h_detail.latitude else "No GPS Data"
                    st.write(f"**Coordinates:** {loc_str}")
                    
                    # Delete button
                    if st.button("🗑️ Delete this Entry", key=f"del_{h_detail.id}"):
                        if history_mgr.delete_entry(h_detail.id):
                            st.success("Entry deleted. Refreshing...")
                            st.rerun()
                        else:
                            st.error("Failed to delete entry.")

    # Summary stats
    st.markdown("---")
    st.markdown("### 📈 Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyses",  len(history))
    col2.metric("Critical", sum(1 for h in history if h.severity_level == "CRITICAL"))
    col3.metric("High",     sum(1 for h in history if h.severity_level == "HIGH"))
    col4.metric(
        "Total Estimated Cost",
        f"Rp {sum(h.repair_cost_idr or 0 for h in history):,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Map
# ══════════════════════════════════════════════════════════════════════════════
def page_map():
    st.markdown('<h1 class="premium-header">Live Map</h1>', unsafe_allow_html=True)

    history      = history_mgr.get_all_history()
    valid_coords = [h for h in history if h.latitude and h.longitude]

    if not valid_coords:
        st.warning(
            "No GPS data found in history. "
            "Upload photos with GPS EXIF metadata to display markers on the map."
        )
        m = folium.Map(location=[-6.2088, 106.8456], zoom_start=12)
    else:
        m = folium.Map(
            location=[valid_coords[0].latitude, valid_coords[0].longitude],
            zoom_start=15
        )
        color_map = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "blue", "LOW": "green"}
        for h in valid_coords:
            color = color_map.get(h.severity_level, "gray")
            popup_html = (
                f"<b>{h.image_name}</b><br>"
                f"Severity: {h.severity_level}<br>"
                f"Volume: {h.volume_cm3:.1f} cm³<br>"
                f"Cost: Rp {h.repair_cost_idr:,.0f}"
            )
            folium.Marker(
                [h.latitude, h.longitude],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=h.image_name,
                icon=folium.Icon(color=color, icon="exclamation-sign")
            ).add_to(m)

    st_folium(m, width=1200, height=600)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown('<h2 style="color:#60a5fa;font-weight:800">InfraSight</h2>',
                    unsafe_allow_html=True)
        st.markdown("*AI-Powered Pothole Analytics*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📷 Analyze", "📋 History", "🗺️ Damage Map"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.info(
            "**How to Use:**\n\n"
            "1. Upload pothole photos. A card or coin must lie flat next to the pothole, "
            "fully visible. Every measurement in cm is derived from it.\n"
            "2. Press **START BATCH ANALYSIS**.\n"
            "3. View 3D visualization, metrics, and repair recommendations per pothole.\n"
            "4. Download the PDF report."
        )

        if _app_config().get("ephemeral_storage"):
            st.caption(
                "Demo instance: analysis history and saved images are wiped whenever "
                "the server restarts. Download the PDF or CSV to keep a result."
            )

    if page == "📷 Analyze":
        page_analyze()
    elif page == "📋 History":
        page_history()
    elif page == "🗺️ Damage Map":
        page_map()


if __name__ == "__main__":
    main()