import os
import time
from datetime import date, datetime
import streamlit as st
from PIL import Image

from src.schemas.models import LivenessStatus, RiskLevel, ApplicationStatus
from src.orchestrator import run_kyc_pipeline
from src.utils.helpers import get_mock_ml_flag

# 1. Page Config and Styling
st.set_page_config(
    page_title="Agentic KYC Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Gradient banner header */
    .header-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        color: white !important;
    }
    .header-banner p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    /* Sleek card components */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .risk-banner-low {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1.5px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: #d1fae5;
    }
    .risk-banner-medium {
        background-color: rgba(245, 158, 11, 0.12);
        border: 1.5px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: #fef3c7;
    }
    .risk-banner-high {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1.5px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: #fee2e2;
    }
    /* Button styles */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# Create uploads directory if not present
os.makedirs("uploads", exist_ok=True)

# 2. Session State Initialization
if "applications" not in st.session_state:
    st.session_state.applications = {}
if "selected_app_id" not in st.session_state:
    st.session_state.selected_app_id = None

# Pre-populate session state with some mock applicants for instant testing
if not st.session_state.applications:
    # Alice Smith - Clean Low Risk
    st.session_state.applications["APP-1001"] = {
        "id": "APP-1001",
        "name": "Alice Smith",
        "id_image": "alice_smith_card.jpg",
        "video": "alice_smith_live.mp4",
        "expected_gesture": "2_fingers_near_eye",
        "created_at": "2026-06-13 14:23:10",
        "status": ApplicationStatus.PENDING,
        "pipeline_run": None
    }
    # Jane Doe - High Risk Watchlist
    st.session_state.applications["APP-1002"] = {
        "id": "APP-1002",
        "name": "Jane Doe",
        "id_image": "jane_doe_id.jpg",
        "video": "jane_doe_live.mp4",
        "expected_gesture": "3_fingers_near_cheek",
        "created_at": "2026-06-13 15:10:45",
        "status": ApplicationStatus.PENDING,
        "pipeline_run": None
    }
    # John Doe - High Risk Liveness Spoof (Physical Spoof)
    st.session_state.applications["APP-1003"] = {
        "id": "APP-1003",
        "name": "John Doe",
        "id_image": "john_doe_card.png",
        "video": "john_doe_spoof.mp4",
        "expected_gesture": "2_fingers_near_eye",
        "created_at": "2026-06-13 16:05:00",
        "status": ApplicationStatus.PENDING,
        "pipeline_run": None
    }
    # Bob Miller - Failed Gesture Challenge
    st.session_state.applications["APP-1004"] = {
        "id": "APP-1004",
        "name": "Bob Miller",
        "id_image": "bob_miller_card.jpg",
        "video": "bob_miller_wrong_gesture.mp4",
        "expected_gesture": "2_fingers_near_eye",
        "created_at": "2026-06-14 09:12:00",
        "status": ApplicationStatus.PENDING,
        "pipeline_run": None
    }
    # Charlie Davis - Deepfake/AI Video Spoof
    st.session_state.applications["APP-1005"] = {
        "id": "APP-1005",
        "name": "Charlie Davis",
        "id_image": "charlie_davis_card.jpg",
        "video": "charlie_davis_deepfake_spoof.mp4",
        "expected_gesture": "1_finger_pointing_to_nose",
        "created_at": "2026-06-14 09:15:00",
        "status": ApplicationStatus.PENDING,
        "pipeline_run": None
    }

# 3. Sidebar Panel
st.sidebar.markdown("### 🛡️ Agentic KYC Platform")
st.sidebar.info(
    f"**Execution Mode**: {'Mock Local (Lightweight)' if get_mock_ml_flag() else 'GPU/ROCm Accelerated'}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Applications Queue")

# Active App Selection
app_options = {
    app_id: f"{app_id} - {app['name']} ({app['status'].value})"
    for app_id, app in st.session_state.applications.items()
}

selected_id = st.sidebar.selectbox(
    "Select Applicant to Review",
    options=list(app_options.keys()),
    format_func=lambda x: app_options[x]
)
st.session_state.selected_app_id = selected_id

st.sidebar.markdown("---")
st.sidebar.markdown("### ➕ Create New Application")
new_id = f"APP-{1000 + len(st.session_state.applications) + 1}"
uploaded_name = st.sidebar.text_input("Applicant Name", placeholder="e.g., Robert Vance")

# Scan uploads directory for files uploaded via JupyterLab file browser
local_files = ["-- Select --"]
if os.path.exists("uploads"):
    local_files += sorted([f for f in os.listdir("uploads") if os.path.isfile(os.path.join("uploads", f))])

st.sidebar.markdown("**ID Document (Image)**")
id_mode = st.sidebar.radio("ID Input Mode", ["Upload File", "Choose from uploads/ folder"], key="id_mode")
uploaded_id_img = None
selected_id_path = None
if id_mode == "Upload File":
    uploaded_id_img = st.sidebar.file_uploader("Upload ID Document", type=["jpg", "jpeg", "png"], key="id_upload")
else:
    selected_id_file = st.sidebar.selectbox("Choose ID File", options=local_files, key="sel_id")
    if selected_id_file != "-- Select --":
        selected_id_path = os.path.join("uploads", selected_id_file)

st.sidebar.markdown("**Face Verification Video**")
vid_mode = st.sidebar.radio("Video Input Mode", ["Upload File", "Choose from uploads/ folder"], key="vid_mode")
uploaded_vid = None
selected_vid_path = None
if vid_mode == "Upload File":
    uploaded_vid = st.sidebar.file_uploader("Upload Face Verification Video", type=["mp4"], key="vid_upload")
else:
    selected_vid_file = st.sidebar.selectbox("Choose Video File", options=local_files, key="sel_vid")
    if selected_vid_file != "-- Select --":
        selected_vid_path = os.path.join("uploads", selected_vid_file)

uploaded_gesture = st.sidebar.selectbox(
    "Onboarding Gesture Challenge",
    options=[
        "2_fingers_near_eye",
        "3_fingers_near_cheek",
        "1_finger_pointing_to_nose"
    ],
    format_func=lambda x: x.replace("_", " ").title()
)

use_minifasnet = st.sidebar.checkbox(
    "Use MiniFASNet (Edge Spoof Model)",
    value=False,
    key="use_minifasnet_cb",
    help="Enable edge-friendly SilentFaceAntiSpoofing (MiniFASNet) model inference fallback."
)

if st.sidebar.button("Add to Queue"):
    has_id = (id_mode == "Upload File" and uploaded_id_img is not None) or (id_mode == "Choose from uploads/ folder" and selected_id_path is not None)
    has_vid = (vid_mode == "Upload File" and uploaded_vid is not None) or (vid_mode == "Choose from uploads/ folder" and selected_vid_path is not None)
    
    if not uploaded_name:
        st.sidebar.error("Please enter applicant name.")
    elif not has_id or not has_vid:
        st.sidebar.error("Please provide both ID document and video using the selected modes.")
    else:
        # Resolve paths
        if id_mode == "Upload File":
            id_path = os.path.join("uploads", f"{new_id}_id_{uploaded_id_img.name}")
            with open(id_path, "wb") as f:
                f.write(uploaded_id_img.read())
        else:
            id_path = selected_id_path

        if vid_mode == "Upload File":
            vid_path = os.path.join("uploads", f"{new_id}_vid_{uploaded_vid.name}")
            with open(vid_path, "wb") as f:
                f.write(uploaded_vid.read())
        else:
            vid_path = selected_vid_path
            
        st.session_state.applications[new_id] = {
            "id": new_id,
            "name": uploaded_name,
            "id_image": id_path,
            "video": vid_path,
            "expected_gesture": uploaded_gesture,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": ApplicationStatus.PENDING,
            "pipeline_run": None
        }
        st.sidebar.success(f"Added {new_id} successfully!")
        st.session_state.selected_app_id = new_id
        st.rerun()

# 4. Main Panel
st.markdown("""
<div class="header-banner">
    <h1>Agentic KYC Intelligence Platform</h1>
    <p>Consolidated Multi-Agent Onboarding Verification System targeting AMD ROCm execution</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.selected_app_id:
    app_id = st.session_state.selected_app_id
    app = st.session_state.applications[app_id]
    
    st.markdown(f"### Reviewing: **{app['name']}** (ID: `{app['id']}`)")
    st.caption(f"Created on: {app['created_at']} | Status: `{app['status'].value}`")
    
    # Display the onboarding gesture challenge assigned to the applicant
    st.markdown(f"""
    <div style="background-color: rgba(79, 70, 229, 0.1); border-left: 5px solid #4f46e5; border-radius: 6px; padding: 1rem; margin-top: 1rem; margin-bottom: 1.5rem;">
        <h4 style="margin: 0; color: #4f46e5; font-size: 1rem;">🎯 Required Onboarding Gesture Challenge</h4>
        <p style="margin: 0.25rem 0 0 0; font-size: 1.15rem; font-weight: 600; color: #1e1b4b;">
            {app.get('expected_gesture', '2_fingers_near_eye').replace('_', ' ').title()}
        </p>
        <small style="color: #6b7280;">Applicant must display this gesture near their face to pass the digital deepfake & spoof verification.</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Run pipeline button
    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Run KYC Pipeline", use_container_width=True)
    with col2:
        if app["pipeline_run"]:
            st.success("Pipeline results loaded from previous run.")
        else:
            st.warning("Pipeline has not been executed for this applicant yet.")

    # Execution trigger
    if run_btn:
        with st.spinner("Executing stateless multi-agent pipeline in parallel..."):
            try:
                ext_res, live_res, screen_res, risk_res = run_kyc_pipeline(
                    app["id_image"],
                    app["video"],
                    app.get("expected_gesture", "2_fingers_near_eye"),
                    app["name"],
                    use_minifasnet=st.session_state.get("use_minifasnet_cb", False)
                )
                app["pipeline_run"] = {
                    "extraction": ext_res,
                    "liveness": live_res,
                    "screening": screen_res,
                    "risk": risk_res
                }
                st.success("Stateless multi-agent pipeline execution completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline execution failed: {str(e)}")

    # Display results
    if app["pipeline_run"]:
        results = app["pipeline_run"]
        ext = results["extraction"]
        live = results["liveness"]
        screen = results["screening"]
        risk = results["risk"]
        
        # Risk Score Callout Banner
        banner_class = "risk-banner-low"
        risk_color = "#10b981"
        if risk.risk_level == RiskLevel.HIGH:
            banner_class = "risk-banner-high"
            risk_color = "#ef4444"
        elif risk.risk_level == RiskLevel.MEDIUM:
            banner_class = "risk-banner-medium"
            risk_color = "#f59e0b"
            
        st.markdown(f"""
        <div class="{banner_class}">
            <h3 style='margin-top: 0; color: inherit;'>Consolidated Risk Report: <span style='color: {risk_color}; font-weight: 800;'>{risk.risk_level.value} RISK</span></h3>
            <p style='font-size: 1.15rem; margin-bottom: 0.5rem;'><strong>Consolidated Risk Score:</strong> <span style='font-size: 1.3rem; font-weight: 800;'>{risk.risk_score} / 100</span></p>
            <p style='margin: 0;'><strong>Explanation:</strong> {risk.explanation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Grid of Agents
        ag_col1, ag_col2, ag_col3 = st.columns(3)
        
        with ag_col1:
            st.markdown("### 📄 Extraction Agent")
            
            ai_check_badge = "✅ CLEAN"
            if ext.ai_generated_check == "AI_GENERATED":
                ai_check_badge = "🚨 AI GENERATED"
            elif ext.ai_generated_check == "SUSPICIOUS":
                ai_check_badge = "⚠️ SUSPICIOUS"

            forgery_badge = "🚨 FORGERY DETECTED" if ext.forgery_detected else "✅ VERIFIED GENUINE"
            
            legibility_status = "✅ SHARP" if ext.legibility_score >= 0.70 else "❌ BLURRY"
            syntax_status = "✅ VALID" if ext.syntax_valid else "❌ INVALID"
            ovi_status = "✅ DETECTED" if ext.ovi_crest_detected else "❌ MISSING"

            ocr_engine = "EasyOCR (Local Fallback)" if getattr(ext, "local_ocr_active", False) else "Qwen2-VL (vLLM Cloud)"
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Extracted Name:</strong> {ext.name}</p>
                <p><strong>DOB:</strong> {ext.dob}</p>
                <p><strong>ID Number:</strong> <code>{ext.id_number}</code></p>
                <p><strong>Extraction Confidence:</strong> {ext.confidence * 100:.1f}%</p>
                <p><strong>Active OCR Engine:</strong> <code>{ocr_engine}</code></p>
                <p><strong>Doc Authenticity:</strong> {forgery_badge}</p>
                <p><strong>AI Generation Check:</strong> {ai_check_badge}</p>
                {"<p style='color: #ef4444; font-size: 0.85rem; margin-top: 0.5rem; margin-bottom: 0;'><strong>Reason:</strong> " + ext.forgery_reason + "</p>" if ext.forgery_detected else ""}
            </div>
            
            <div class="metric-card" style="margin-top: 1rem; border-left: 4px solid #6366f1;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #818cf8;">Programmatic Security Checks</h4>
                <p><strong>Legibility Score:</strong> {ext.legibility_score:.2f} ({legibility_status})</p>
                <p><strong>ID Syntax Match:</strong> {syntax_status}</p>
                <p><strong>OVI Hologram Crest:</strong> {ovi_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show simulated or uploaded image if path exists
            if os.path.exists(app["id_image"]):
                try:
                    img = Image.open(app["id_image"])
                    st.image(img, caption="ID Document Upload", use_container_width=True)
                except Exception:
                    st.caption("ID Document preview unavailable.")
            else:
                # Preloaded mock sample visual fallback
                st.caption(f"📁 Document File: `{app['id_image']}` (Mock Preview)")
                
        with ag_col2:
            st.markdown("### 🎥 Liveness Agent")
            liveness_badge = "🟢 PASSED" if live.liveness_status == LivenessStatus.PASSED else "🔴 FAILED"
            
            # Sub-check status
            physical_status = "❌ FAILED" if live.physical_spoof_detected else "✅ PASSED"
            gesture_status = "✅ PASSED" if live.gestural_challenge_passed else "❌ FAILED"
            deepfake_status = "❌ FAILED" if live.digital_deepfake_detected else "✅ PASSED"
            
            face_match_decision = getattr(live, "face_match_decision", "MATCH")
            face_match_badge = "✅ MATCH" if face_match_decision == "MATCH" else "🚨 MISMATCH"
            face_sim = getattr(live, "face_similarity_score", 1.0)
            liveness_model = "MiniFASNet (Edge)" if getattr(live, "minifasnet_active", False) else "MobileNetV3 (Standard)"

            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Liveness Outcome:</strong> {liveness_badge}</p>
                <p><strong>Face Verification:</strong> {face_match_badge}</p>
                <p><strong>Face Similarity Score:</strong> {face_sim * 100:.1f}%</p>
                <p><strong>Active Liveness Model:</strong> <code>{liveness_model}</code></p>
                <p><strong>Physical Liveness:</strong> {physical_status}</p>
                <p><strong>Gesture Verification:</strong> {gesture_status}</p>
                <p><strong>Deepfake/AI Detection:</strong> {deepfake_status}</p>
                <p><strong>Inference Confidence:</strong> {live.confidence * 100:.1f}%</p>
                <p><strong>Spoof Probability:</strong> {live.spoof_probability * 100:.1f}%</p>
                <p><strong>Security Flags:</strong> {', '.join(live.flags) if live.flags else 'None'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mathematical Telemetry Plots
            fft_val = live.fft_metrics.get("peak_ratio", 1.4)
            fft_label = "GRID DETECTED" if live.fft_grid_detected else "NORMAL"
            
            flow_val = live.optical_flow_metrics.get("variance", 0.12)
            flow_label = "WARPING" if live.optical_flow_mismatch else "NORMAL"

            st.markdown("#### Mathematical Telemetry")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(
                    label="FFT Peak Ratio",
                    value=f"{fft_val:.2f}",
                    delta=fft_label,
                    delta_color="inverse" if live.fft_grid_detected else "normal"
                )
            with m_col2:
                st.metric(
                    label="Optical Flow Var",
                    value=f"{flow_val:.2f}",
                    delta=flow_label,
                    delta_color="inverse" if live.optical_flow_mismatch else "normal"
                )
            
            if live.rppg_signal:
                st.markdown("##### Green-Channel Cardiac rPPG Signal")
                st.line_chart(live.rppg_signal, height=130)
                rppg_pulse_label = "Heartbeat Rhythm Detected" if live.rppg_pulse_detected else "Flatline / No Pulse Wave"
                st.caption(f"rPPG Signal Rhythm: **{rppg_pulse_label}**")
                
            st.caption(f"📁 Face Video File: `{app['video']}`")
            
        with ag_col3:
            st.markdown("### 🔍 Screening Agent")
            screen_badge = "🚨 Watchlist Hits Found" if screen.match_found else "✅ No Watchlist Matches"
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Screening Outcome:</strong> {screen_badge}</p>
                <p><strong>Assigned Risk level:</strong> {screen.risk_level.value}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if screen.watchlist_hits:
                st.markdown("**Watchlist Matches:**")
                for hit in screen.watchlist_hits:
                    matched_lbl = "ID Document" if hit.matched_on == "extracted_name" else "submitted name"
                    st.error(f"⚠️ **{hit.list_name}**: {hit.reason} (Match Score: {hit.match_score * 100:.1f}% | Matched on {matched_lbl}: {hit.name})")
            if screen.adverse_media_hits:
                st.markdown("**Adverse Media Hits:**")
                for hit in screen.adverse_media_hits:
                    matched_lbl = "ID Document" if hit.matched_on == "extracted_name" else "submitted name"
                    st.warning(f"📰 **{hit.source}**: *{hit.title}* ({hit.sentiment} | Matched on {matched_lbl})")
                    if hit.url:
                        st.caption(f"[Link to Article]({hit.url})")
                        
        st.markdown("---")
        
        # Audit logs tab / table
        st.markdown("### 🛡️ Pipeline Audit Logs & Traceability")
        
        audit_data = []
        for agent_name, entry in risk.agent_audit_log.items():
            audit_data.append({
                "Agent": agent_name,
                "Status": entry["status"],
                "Latency (s)": f"{entry['latency_seconds']:.3f}s",
                "Model / Engine": entry["model_used"],
                "Details": str(entry["details"]),
                "Timestamp": entry["timestamp"]
            })
        st.table(audit_data)
        
        # Decision control panel
        st.markdown("### ✍️ Reviewer Action Decision")
        
        dec_col1, dec_col2, dec_col3 = st.columns([1, 1, 3])
        with dec_col1:
            approve_btn = st.button("✅ Approve Application", use_container_width=True, type="primary")
        with dec_col2:
            escalate_btn = st.button("🚨 Escalate Application", use_container_width=True)
            
        if approve_btn:
            app["status"] = ApplicationStatus.APPROVED
            st.success(f"Application {app_id} has been APPROVED.")
            time.sleep(1)
            st.rerun()
            
        if escalate_btn:
            app["status"] = ApplicationStatus.ESCALATED
            st.info(f"Application {app_id} has been ESCALATED.")
            time.sleep(1)
            st.rerun()
else:
    st.info("Please select or add an application from the sidebar to begin reviewer validation.")
