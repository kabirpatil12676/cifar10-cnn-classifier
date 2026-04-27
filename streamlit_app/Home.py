"""
Home.py  —  Entry point for the CIFAR-10 CNN Classifier Streamlit app.
Renamed from app.py so the sidebar shows "Home" instead of "app".

Streamlit Cloud deployment:  set Main file path → streamlit_app/Home.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="assets/favicon.png" if (ROOT / "assets" / "favicon.png").exists() else "🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/kabirpatil12676/cifar10-cnn-classifier",
        "Report a bug": "https://github.com/kabirpatil12676/cifar10-cnn-classifier/issues",
        "About": "CIFAR-10 CNN Classifier — Production PyTorch demo by Kabir Patil.",
    },
)

from utils.model_loader import load_model  # noqa: E402


@st.cache_resource(show_spinner="Loading model weights…")
def _get_model():
    return load_model(ROOT / "checkpoints" / "best_model.pth")


model, model_meta = _get_model()

if "model" not in st.session_state:
    st.session_state["model"] = model
if "model_meta" not in st.session_state:
    st.session_state["model_meta"] = model_meta

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #4F8EF7 0%, #A78BFA 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem;
}
.hero-sub {
    font-size: 1.05rem; color: #6B7280; margin-bottom: 1.8rem; font-weight: 400;
}
.stat-card {
    background: #161B27; border: 1px solid #1F2937;
    border-radius: 10px; padding: 1.25rem 1.5rem; text-align: center;
}
.stat-value { font-size: 1.6rem; font-weight: 700; color: #4F8EF7; }
.stat-label { font-size: 0.8rem; color: #6B7280; margin-top: 0.2rem; letter-spacing: 0.04em; text-transform: uppercase; }
.section-divider { border: none; border-top: 1px solid #1F2937; margin: 1.8rem 0; }
.arch-block {
    background: #0D1117; border: 1px solid #1F2937; border-radius: 8px;
    padding: 1rem 1.4rem; font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.78rem; color: #8B9CF4; white-space: pre; overflow-x: auto;
}
.badge-row a { text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">CIFAR-10 CNN Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Production-grade PyTorch CNN with ResNet-style skip connections — '
    '90.22% test accuracy on CIFAR-10.</p>',
    unsafe_allow_html=True,
)

col_gh, col_li, col_nb, _ = st.columns([1, 1, 1.2, 5])
col_gh.link_button("GitHub", "https://github.com/kabirpatil12676/cifar10-cnn-classifier")
col_li.link_button("LinkedIn", "https://www.linkedin.com/in/kabir-patil-7a2a9b30b/")
col_nb.link_button("View Notebook", "https://github.com/kabirpatil12676/cifar10-cnn-classifier/blob/main/notebooks/CIFAR10_Analysis.ipynb")

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# ── Model status ──────────────────────────────────────────────────────────────
if model_meta.get("is_fallback"):
    st.warning(
        "Checkpoint not found. Running with randomly-initialized weights — "
        "place `best_model.pth` in `checkpoints/` and restart.",
        icon="⚠️",
    )
else:
    val_acc = model_meta.get("val_acc", 0)
    try:
        val_acc_str = f"{float(val_acc):.2f}%"
    except (TypeError, ValueError):
        val_acc_str = "N/A"
    st.success(
        f"Model loaded — val accuracy: **{val_acc_str}** "
        f"(epoch {model_meta.get('epoch', '?')})",
        icon="✅",
    )

# ── Key stats ────────────────────────────────────────────────────────────────
st.subheader("Performance Summary")
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, "90.22%",  "Test Accuracy"),
    (c2, "~2.8M",   "Parameters"),
    (c3, "38",      "Training Epochs"),
    (c4, "10",      "Object Classes"),
]:
    col.markdown(
        f'<div class="stat-card"><div class="stat-value">{val}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# ── Architecture ─────────────────────────────────────────────────────────────
st.subheader("Model Architecture")
arch = """\
Input  (3 x 32 x 32)
  |
  v  Block 1  ResidualBlock  3 -> 64      MaxPool2d(2)   -> (64 x 16 x 16)
  v  Block 2  ResidualBlock  64 -> 128    MaxPool2d(2)   -> (128 x 8 x 8)
  v  Block 3  ResidualBlock  128 -> 256   AvgPool2d(2)   -> (256 x 2 x 2)
  |
  v  Head :  Flatten -> Linear(1024->512) -> ReLU -> Dropout(0.4) -> Linear(512->10)
  |
  v  Logits (10 classes)"""
st.markdown(f'<div class="arch-block">{arch}</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# ── Pages ────────────────────────────────────────────────────────────────────
st.subheader("Navigate")
pa, pb, pc, pd, pe = st.columns(5)
pa.page_link("pages/1_Live_Prediction.py",  label="Live Prediction")
pb.page_link("pages/2_GradCAM_Explorer.py", label="GradCAM Explorer")
pc.page_link("pages/3_Model_Report.py",     label="Model Report")
pd.page_link("pages/4_Dataset_Explorer.py", label="Dataset Explorer")
pe.page_link("pages/5_Batch_Inference.py",  label="Batch Inference")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### CIFAR-10 Classifier")
    st.caption("Kabir Patil · ResNet CNN · PyTorch")
    st.divider()
    st.markdown(
        "**Classes**  \nairplane · automobile · bird · cat · deer  \n"
        "dog · frog · horse · ship · truck"
    )
    st.divider()
    st.caption("Built with PyTorch + Streamlit")
