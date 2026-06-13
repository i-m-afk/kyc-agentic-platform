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
        "created_at": "2026-06-13 15:10:45",
        "status": ApplicationStatus.PENDING,
        "pipeline_run": None
    }
    # John Doe - High Risk Liveness Spoof
    st.session_state.applications["APP-1003"] = {
        "id": "APP-1003",
        "name": "John Doe",
        "id_image": "john_doe_card.png",
        "video": "john_doe_spoof.mp4",
        "created_at": "2026-06-13 16:05:00",
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
uploaded_id_img = st.sidebar.file_uploader("Upload ID Document (Image)", type=["jpg", "jpeg", "png"])
uploaded_vid = st.sidebar.file_uploader("Upload Face Verification Video", type=["mp4"])

if st.sidebar.button("Add to Queue"):
    if not uploaded_name:
        st.sidebar.error("Please enter applicant name.")
    elif not uploaded_id_img or not uploaded_vid:
        st.sidebar.error("Please upload both ID document and video.")
    else:
        # Save uploads
        id_path = os.path.join("uploads", f"{new_id}_id_{uploaded_id_img.name}")
        vid_path = os.path.join("uploads", f"{new_id}_vid_{uploaded_vid.name}")
        
        with open(id_path, "wb") as f:
            f.write(uploaded_id_img.read())
        with open(vid_path, "wb") as f:
            f.write(uploaded_vid.read())
            
        st.session_state.applications[new_id] = {
            "id": new_id,
            "name": uploaded_name,
            "id_image": id_path,
            "video": vid_path,
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
                    app["video"]
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
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Extracted Name:</strong> {ext.name}</p>
                <p><strong>DOB:</strong> {ext.dob}</p>
                <p><strong>ID Number:</strong> <code>{ext.id_number}</code></p>
                <p><strong>Extraction Confidence:</strong> {ext.confidence * 100:.1f}%</p>
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
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Liveness Status:</strong> {liveness_badge}</p>
                <p><strong>Inference Confidence:</strong> {live.confidence * 100:.1f}%</p>
                <p><strong>Spoof Probability:</strong> {live.spoof_probability * 100:.1f}%</p>
                <p><strong>Security Flags:</strong> {', '.join(live.flags) if live.flags else 'None'}</p>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"📁 Face Video File: `{app['video']}` (Mock Preview)")
            
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
                    st.error(f"⚠️ **{hit.list_name}**: {hit.reason} (Match Score: {hit.match_score * 100:.1f}%)")
            if screen.adverse_media_hits:
                st.markdown("**Adverse Media Hits:**")
                for hit in screen.adverse_media_hits:
                    st.warning(f"📰 **{hit.source}**: *{hit.title}* ({hit.sentiment})")
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
