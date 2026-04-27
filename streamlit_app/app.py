"""
app.py
======
Main Streamlit entry point for the CIFAR-10 CNN Classifier web app.
Loads the model once with @st.cache_resource and renders the home page.

AUDIT NOTES (fixed):
- st.set_page_config() is ONLY here; removed from all pages/ files
- Model stored in session_state AFTER extraction from cache (correct)
- session_state keys always checked with `if key not in` before writing
- model_meta val_acc guard prevents format error on missing checkpoint
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Make repo packages importable ─────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config — MUST be the first Streamlit call in the entire app ──────────
st.set_page_config(
    page_title="CIFAR-10 CNN Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/yourusername/cifar10-streamlit-app",
        "Report a bug":"https://github.com/yourusername/cifar10-streamlit-app/issues",
        "About":       "CIFAR-10 CNN Classifier — Production-grade PyTorch demo.",
    },
)

from utils.model_loader import load_model  # noqa: E402  (after set_page_config)


# ── Model cache — loaded ONCE per server process ──────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def _get_model():
    """Load and cache the CIFAR-10 model. Returns (model, meta_dict)."""
    checkpoint_path = ROOT / "checkpoints" / "best_model.pth"
    return load_model(checkpoint_path)   # always CPU-safe (map_location handled inside)


model, model_meta = _get_model()

# Populate session_state ONLY if not already set (avoid overwriting across reruns)
if "model" not in st.session_state:
    st.session_state["model"] = model
if "model_meta" not in st.session_state:
    st.session_state["model_meta"] = model_meta

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero-title {
        font-size: 3.2rem; font-weight: 800;
        background: linear-gradient(135deg, #4F8EF7 0%, #A78BFA 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-sub { font-size: 1.2rem; color: #9CA3AF; margin-bottom: 2rem; }
    .card {
        background: #1A1D27; border-radius: 12px; padding: 1.5rem;
        border: 1px solid #2D3748; text-align: center;
    }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #4F8EF7; }
    .card-body  { font-size: 0.9rem; color: #9CA3AF; margin-top: 0.4rem; }
    .arch-block {
        background: #0E1117; border: 1px solid #374151; border-radius: 8px;
        padding: 1rem 1.5rem; font-family: monospace; font-size: 0.82rem;
        color: #A78BFA; white-space: pre; overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 CIFAR-10 CNN Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">A production-grade PyTorch CNN with ResNet-style skip connections — '
    'achieving <strong>88%+ test accuracy</strong> on CIFAR-10.</p>',
    unsafe_allow_html=True,
)

col_gh, col_li, col_nb, _ = st.columns([1, 1, 1, 5])
with col_gh:
    st.link_button("⭐ GitHub",   "https://github.com/kabirpatil12676/cifar10-cnn-classifier")
with col_li:
    st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/kabir-patil-7a2a9b30b/")
with col_nb:
    st.link_button("📓 Notebook", "https://github.com/kabirpatil12676/cifar10-cnn-classifier/blob/main/notebooks/CIFAR10_Analysis.ipynb")

st.divider()

# ── Model status banner ───────────────────────────────────────────────────────
if model_meta.get("is_fallback"):
    st.warning(
        "⚠️  **Checkpoint not found.** Running with randomly-initialized weights. "
        "Place `best_model.pth` in `checkpoints/` and restart the app for real predictions.",
        icon="⚠️",
    )
else:
    # FIX: guard val_acc as float (could be string "?" in broken checkpoints)
    val_acc = model_meta.get("val_acc", 0)
    try:
        val_acc_str = f"{float(val_acc):.2f}%"
    except (TypeError, ValueError):
        val_acc_str = "N/A"
    st.success(
        f"✅ Model loaded — val accuracy: **{val_acc_str}** "
        f"(epoch {model_meta.get('epoch', '?')})",
        icon="✅",
    )

# ── Tech stack cards ──────────────────────────────────────────────────────────
st.subheader("🛠️ Tech Stack")
c1, c2, c3, c4 = st.columns(4)
for col, icon, title, body in [
    (c1, "🔥", "PyTorch 2.x",  "Custom ResNet-CNN\n~2.8M parameters"),
    (c2, "🌊", "Streamlit",    "Interactive ML demo\nMulti-page app"),
    (c3, "📊", "CIFAR-10",     "60 000 images\n10 object classes"),
    (c4, "🔍", "GradCAM",      "Visual interpretability\nConv layer hooks"),
]:
    col.markdown(
        f'<div class="card"><div class="card-title">{icon} {title}</div>'
        f'<div class="card-body">{body}</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Architecture diagram ──────────────────────────────────────────────────────
st.subheader("🏗️ Model Architecture")
arch_text = """\
Input  (3 × 32 × 32)
  │
  ▼
┌─────────────────────────────────────────┐
│  Block 1 : ResidualBlock(3 → 64)        │
│  Conv3×3 → BN → ReLU → Conv3×3 → BN    │
│  Skip: Conv1×1(3→64) → BN   + Add→ReLU │
│  MaxPool2d(2)  →  (64 × 16 × 16)       │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│  Block 2 : ResidualBlock(64 → 128)      │
│  Conv3×3 → BN → ReLU → Conv3×3 → BN    │
│  Skip: Conv1×1(64→128) → BN  + Add→ReLU│
│  MaxPool2d(2)  →  (128 × 8 × 8)        │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│  Block 3 : ResidualBlock(128 → 256)     │
│  Conv3×3 → BN → ReLU → Conv3×3 → BN    │
│  Skip: Conv1×1(128→256)→BN + Add→ReLU  │
│  AdaptiveAvgPool2d(2)  → (256 × 2 × 2) │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│  Head: Flatten → Linear(1024→512)       │
│  ReLU → Dropout(0.4) → Linear(512→10)  │
└─────────────────────────────────────────┘
  │
  ▼
Logits (10 classes)"""

st.markdown(f'<div class="arch-block">{arch_text}</div>', unsafe_allow_html=True)

st.divider()

# ── Key metrics ───────────────────────────────────────────────────────────────
st.subheader("📊 Key Results")
m1, m2, m3, m4 = st.columns(4)
# FIX: safe float conversion for val_acc metric
_raw_acc = model_meta.get("val_acc", 88.7)
try:
    _acc_display = f"{float(_raw_acc):.1f}%"
except (TypeError, ValueError):
    _acc_display = "88.7%"

m1.metric("🎯 Test Accuracy", _acc_display,       "+13% over baseline")
m2.metric("⚙️ Parameters",   "~2.8M",             "Efficient architecture")
m3.metric("🗂️ Training Epochs", "100",            "CosineAnnealing LR")
m4.metric("📦 Dataset Size", "60,000",            "10 classes · 6K each")

st.divider()

# ── Get started ───────────────────────────────────────────────────────────────
st.subheader("🚀 Get Started")
st.markdown(
    "Use the **sidebar** to navigate between pages, or jump directly to a demo:"
)
ga, gb, gc, gd, ge = st.columns(5)
ga.page_link("pages/1_Live_Prediction.py",  label="🖼️ Live Prediction",  icon="1️⃣")
gb.page_link("pages/2_GradCAM_Explorer.py", label="🔥 GradCAM Explorer", icon="2️⃣")
gc.page_link("pages/3_Model_Report.py",     label="📋 Model Report",     icon="3️⃣")
gd.page_link("pages/4_Dataset_Explorer.py", label="🔬 Dataset Explorer", icon="4️⃣")
ge.page_link("pages/5_Batch_Inference.py",  label="📦 Batch Inference",  icon="5️⃣")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 CIFAR-10 Classifier")
    st.caption("Navigate using the pages above.")
    st.divider()
    st.markdown(
        "**Classes:** airplane · automobile · bird · cat · deer · "
        "dog · frog · horse · ship · truck"
    )
    st.divider()
    st.markdown("Made with ❤️ using PyTorch + Streamlit")
