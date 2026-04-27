"""
app.py  —  Navigation controller (entry point for Streamlit Cloud).
Uses st.navigation() so the sidebar shows proper page names instead of filenames.
Streamlit Cloud main file path: streamlit_app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# st.set_page_config MUST be first Streamlit call
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com/kabirpatil12676/cifar10-cnn-classifier",
        "Report a bug": "https://github.com/kabirpatil12676/cifar10-cnn-classifier/issues",
        "About":        "CIFAR-10 CNN Classifier — by Kabir Patil",
    },
)

# Load model ONCE here so every page can read from session_state
from utils.model_loader import load_model   # noqa: E402


@st.cache_resource(show_spinner="Loading model…")
def _load():
    return load_model(ROOT / "checkpoints" / "best_model.pth")


model, model_meta = _load()
if "model"      not in st.session_state:
    st.session_state["model"]      = model
if "model_meta" not in st.session_state:
    st.session_state["model_meta"] = model_meta

# ── Navigation (st.navigation controls sidebar labels, not filenames) ─────────
home    = st.Page("Home.py",                          title="Home",             icon=":material/home:",             default=True)
pred    = st.Page("pages/1_Live_Prediction.py",       title="Live Prediction",  icon=":material/image:")
gradcam = st.Page("pages/2_GradCAM_Explorer.py",      title="GradCAM Explorer", icon=":material/visibility:")
report  = st.Page("pages/3_Model_Report.py",          title="Model Report",     icon=":material/bar_chart:")
dataset = st.Page("pages/4_Dataset_Explorer.py",      title="Dataset Explorer", icon=":material/dataset:")
batch   = st.Page("pages/5_Batch_Inference.py",       title="Batch Inference",  icon=":material/batch_prediction:")

pg = st.navigation(
    {
        "": [home],
        "Analysis": [pred, gradcam],
        "Reports":  [report, dataset, batch],
    }
)
pg.run()
