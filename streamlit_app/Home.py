"""
Home.py  —  Home page UI only.
Model loading and st.set_page_config are handled by app.py (entry point).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Model loaded by app.py — read from session_state
if "model" not in st.session_state:
    st.error("Please start the app via app.py so the model can initialise.")
    st.stop()

model      = st.session_state["model"]
model_meta = st.session_state["model_meta"]

# ── CSS ───────────────────────────────────────────────────────────────────────
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
.hero-sub { font-size: 1.05rem; color: #6B7280; margin-bottom: 1.8rem; }
.stat-card {
    background: #161B27; border: 1px solid #1F2937;
    border-radius: 10px; padding: 1.25rem 1.5rem; text-align: center;
}
.stat-value { font-size: 1.6rem; font-weight: 700; color: #4F8EF7; }
.stat-label { font-size: 0.8rem; color: #6B7280; margin-top: 0.2rem;
              letter-spacing: 0.04em; text-transform: uppercase; }
.section-divider { border: none; border-top: 1px solid #1F2937; margin: 1.8rem 0; }
.arch-block {
    background: #0D1117; border: 1px solid #1F2937; border-radius: 8px;
    padding: 1rem 1.4rem; font-family: 'Courier New', monospace;
    font-size: 0.78rem; color: #8B9CF4; white-space: pre; overflow-x: auto;
}
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
        "Checkpoint not found — running with randomly-initialized weights. "
        "Place `best_model.pth` in `checkpoints/` and restart.",
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

# ── Stats ─────────────────────────────────────────────────────────────────────
st.subheader("Performance Summary")
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, "90.22%", "Test Accuracy"),
    (c2, "~2.8M",  "Parameters"),
    (c3, "38",     "Training Epochs"),
    (c4, "10",     "Object Classes"),
]:
    col.markdown(
        f'<div class="stat-card"><div class="stat-value">{val}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# ── Architecture ──────────────────────────────────────────────────────────────
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

# ── Results table ─────────────────────────────────────────────────────────────
st.subheader("Test Set Results")
import pandas as pd
results = [
    ("Airplane",    0.91, 0.90, 0.90),
    ("Automobile",  0.94, 0.94, 0.94),
    ("Bird",        0.88, 0.87, 0.87),
    ("Cat",         0.80, 0.78, 0.79),
    ("Deer",        0.90, 0.91, 0.90),
    ("Dog",         0.84, 0.85, 0.84),
    ("Frog",        0.93, 0.94, 0.93),
    ("Horse",       0.94, 0.93, 0.94),
    ("Ship",        0.94, 0.95, 0.94),
    ("Truck",       0.94, 0.94, 0.94),
]
df = pd.DataFrame(results, columns=["Class", "Precision", "Recall", "F1-Score"])
st.dataframe(df.style.background_gradient(subset=["F1-Score"], cmap="Blues"),
             use_container_width=True, hide_index=True)
st.caption("Overall Test Accuracy: **90.22%**  |  Macro F1: **0.905**")
