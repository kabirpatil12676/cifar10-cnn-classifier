"""
Home.py  —  Premium Home Page for CIFAR-10 CNN Classifier.
Model loading and st.set_page_config are handled by app.py.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if "model" not in st.session_state:
    st.error("Please start the app via app.py.")
    st.stop()

model_meta = st.session_state["model_meta"]

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero */
.hero-wrap {
    background: linear-gradient(135deg, #0f1724 0%, #111827 60%, #0d1528 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 2.5rem 2.8rem 2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(79,142,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -60px; left: 40%;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(167,139,250,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    color: #4F8EF7; text-transform: uppercase; margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 3rem; font-weight: 900; letter-spacing: -1px; line-height: 1.08;
    background: linear-gradient(135deg, #e2e8f0 0%, #a0b4ff 50%, #c4b5fd 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.8rem;
}
.hero-sub {
    font-size: 1rem; color: #64748b; font-weight: 400; max-width: 580px;
    line-height: 1.65; margin-bottom: 1.6rem;
}
.pill {
    display: inline-block;
    background: rgba(79,142,247,0.12); border: 1px solid rgba(79,142,247,0.25);
    color: #93c5fd; border-radius: 999px; padding: 0.25rem 0.85rem;
    font-size: 0.75rem; font-weight: 600; margin-right: 0.5rem; margin-bottom: 0.5rem;
}

/* Stat cards */
.stat-grid { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 130px;
    background: #0f172a; border: 1px solid #1e293b; border-radius: 12px;
    padding: 1.2rem 1.4rem; text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #4F8EF7; }
.stat-value {
    font-size: 2rem; font-weight: 800; line-height: 1;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 0.7rem; color: #475569; margin-top: 0.4rem;
    letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;
}
.stat-sub { font-size: 0.72rem; color: #22c55e; margin-top: 0.25rem; }

/* Section headers */
.section-head {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    margin: 1.8rem 0 0.8rem; display: flex; align-items: center; gap: 0.5rem;
}
.section-head::after {
    content: ''; flex: 1; height: 1px; background: #1e293b; margin-left: 0.6rem;
}

/* Feature cards */
.feature-card {
    background: #0f172a; border: 1px solid #1e293b; border-radius: 12px;
    padding: 1.25rem 1.4rem; height: 100%;
    transition: border-color 0.2s, transform 0.2s;
}
.feature-card:hover { border-color: #4F8EF7; transform: translateY(-2px); }
.feature-icon { font-size: 1.6rem; margin-bottom: 0.6rem; }
.feature-title { font-size: 0.9rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.3rem; }
.feature-desc { font-size: 0.78rem; color: #64748b; line-height: 1.5; }

/* Table */
.real-table { font-size: 0.82rem; }
.divider { border: none; border-top: 1px solid #1e293b; margin: 1.6rem 0; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Deep Learning &nbsp;·&nbsp; Computer Vision &nbsp;·&nbsp; PyTorch</div>
  <div class="hero-title">CIFAR-10 CNN<br>Classifier</div>
  <div class="hero-sub">
    A production-grade ResNet-inspired CNN that classifies 32×32 images
    across 10 object categories — achieving <strong style="color:#60a5fa">90.22% test accuracy</strong>
    with only ~2.8M parameters.
  </div>
  <span class="pill">PyTorch 2.x</span>
  <span class="pill">Residual Blocks</span>
  <span class="pill">GradCAM</span>
  <span class="pill">Label Smoothing</span>
  <span class="pill">Mixed Precision</span>
</div>
""", unsafe_allow_html=True)

# Model status inline (no big banner)
val_acc = model_meta.get("val_acc", 0)
try:
    val_acc_f = float(val_acc)
    status_str = f"✅ &nbsp; Model loaded &nbsp;·&nbsp; Val accuracy <strong>{val_acc_f:.2f}%</strong> &nbsp;·&nbsp; Epoch {model_meta.get('epoch', '?')}"
except Exception:
    status_str = "⚠️ &nbsp; Running with random weights"

st.markdown(
    f'<p style="font-size:0.82rem;color:#475569;margin:-0.8rem 0 1.2rem;">{status_str}</p>',
    unsafe_allow_html=True,
)

col_gh, col_li, col_nb, _ = st.columns([0.9, 0.9, 1.2, 5])
col_gh.link_button("GitHub", "https://github.com/kabirpatil12676/cifar10-cnn-classifier")
col_li.link_button("LinkedIn", "https://www.linkedin.com/in/kabir-patil-7a2a9b30b/")
col_nb.link_button("View Notebook", "https://github.com/kabirpatil12676/cifar10-cnn-classifier/blob/main/notebooks/CIFAR10_Analysis.ipynb")

# ═══════════════════════════════════════════════════════════════════════════════
# STAT CARDS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Performance at a Glance</div>', unsafe_allow_html=True)
st.markdown("""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-value">90.22%</div>
    <div class="stat-label">Test Accuracy</div>
    <div class="stat-sub">+13% vs baseline</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">0.902</div>
    <div class="stat-label">Macro F1</div>
    <div class="stat-sub">Weighted avg</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">~2.8M</div>
    <div class="stat-label">Parameters</div>
    <div class="stat-sub">ResNet-inspired</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">38</div>
    <div class="stat-label">Best Epoch</div>
    <div class="stat-sub">of 40 trained</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">~8ms</div>
    <div class="stat-label">Inference</div>
    <div class="stat-sub">Per image (CPU)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT: Architecture | Per-class results
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-head">Model Architecture</div>', unsafe_allow_html=True)
    st.code("""\
Input  (3 × 32 × 32)
  │
  ├── Block 1  ResidualBlock  3  → 64    + MaxPool → (64 × 16 × 16)
  ├── Block 2  ResidualBlock  64 → 128   + MaxPool → (128 × 8 × 8)
  └── Block 3  ResidualBlock  128 → 256  + AvgPool → (256 × 2 × 2)
  │
  └── Head: Flatten → FC(1024→512) → ReLU → Dropout(0.4) → FC(512→10)
            ↓
        Logits (10 classes)""", language="text")

    st.markdown('<div class="section-head">Training Setup</div>', unsafe_allow_html=True)
    setup = {
        "Optimizer": "AdamW  (lr=1e-3, wd=1e-4)",
        "Scheduler": "CosineAnnealingLR",
        "Loss": "LabelSmoothingCE  (ε=0.1)",
        "Augmentation": "RandomCrop + HFlip + ColorJitter",
        "Mixed Precision": "torch.cuda.amp  (AMP)",
        "Early Stopping": "patience = 15 epochs",
        "Batch Size": "128",
    }
    for k, v in setup.items():
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:0.35rem 0;'
            f'border-bottom:1px solid #1e293b;font-size:0.82rem;">'
            f'<span style="color:#64748b;">{k}</span>'
            f'<span style="color:#cbd5e1;font-weight:500;">{v}</span></div>',
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown('<div class="section-head">Per-Class F1 Scores</div>', unsafe_allow_html=True)

    results = [
        ("Automobile", 0.959),
        ("Ship",        0.942),
        ("Frog",        0.941),
        ("Truck",       0.938),
        ("Horse",       0.914),
        ("Airplane",    0.901),
        ("Deer",        0.897),
        ("Bird",        0.873),
        ("Dog",         0.853),
        ("Cat",         0.803),
    ]

    for cls, f1 in results:
        pct   = int(f1 * 100)
        color = "#22c55e" if f1 >= 0.93 else "#f59e0b" if f1 >= 0.87 else "#f87171"
        bar   = int(f1 * 180)
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.55rem;">
              <span style="width:82px;font-size:0.78rem;color:#94a3b8;text-align:right;">{cls}</span>
              <div style="flex:1;background:#1e293b;border-radius:4px;height:8px;overflow:hidden;">
                <div style="width:{bar}px;max-width:100%;background:{color};
                            height:100%;border-radius:4px;transition:width 0.5s;"></div>
              </div>
              <span style="width:36px;font-size:0.78rem;font-weight:700;color:{color};">{f1:.3f}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="font-size:0.7rem;color:#475569;margin-top:0.6rem;">'
        'Green ≥ 0.93 &nbsp;·&nbsp; Amber ≥ 0.87 &nbsp;·&nbsp; Red &lt; 0.87</p>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 CLASS GALLERY  (real images from dataset)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-head">Dataset — What the Model Sees</div>', unsafe_allow_html=True)
st.caption("One real CIFAR-10 sample per class (32×32 pixels, upscaled for display).")


@st.cache_data(show_spinner="Loading CIFAR-10 samples…")
def _get_one_per_class() -> dict[str, bytes]:
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        cache = Path(tempfile.gettempdir()) / "cifar10_streamlit"
        ds    = CIFAR10(root=str(cache), train=False, download=True,
                        transform=transforms.ToTensor())
        out: dict[str, bytes] = {}
        for img_t, lbl in ds:
            cls = CLASSES[lbl]
            if cls not in out:
                pil = Image.fromarray(
                    (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                ).resize((96, 96), Image.Resampling.NEAREST)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                out[cls] = buf.getvalue()
            if len(out) == 10:
                break
        return out
    except Exception:
        return {}


samples = _get_one_per_class()
if samples:
    cols = st.columns(10)
    for col, cls in zip(cols, CLASSES):
        if cls in samples:
            col.image(samples[cls], caption=cls.capitalize(), width=80)
else:
    st.info("Dataset samples not available offline.", icon="ℹ️")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE CARDS  (what each app page does)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-head">Explore the App</div>', unsafe_allow_html=True)

features = [
    ("Live Prediction",  "Upload any image and get instant top-5 class predictions with confidence scores.",         "pages/1_Live_Prediction.py"),
    ("GradCAM Explorer", "See exactly which pixels drove the model decision — hook-based gradient heatmaps.",       "pages/2_GradCAM_Explorer.py"),
    ("Model Report",     "Full evaluation: confusion matrix, per-class F1, training curves, error analysis.",       "pages/3_Model_Report.py"),
    ("Dataset Explorer", "Browse CIFAR-10 class distributions, sample grids, and augmentation previews.",          "pages/4_Dataset_Explorer.py"),
    ("Batch Inference",  "Upload up to 50 images at once and download all predictions as a CSV file.",             "pages/5_Batch_Inference.py"),
]

fc = st.columns(5)
icons = ["🖼", "👁", "📊", "🔬", "📦"]
for col, (title, desc, page), icon in zip(fc, features, icons):
    with col:
        st.markdown(
            f'<div class="feature-card">'
            f'<div class="feature-icon">{icon}</div>'
            f'<div class="feature-title">{title}</div>'
            f'<div class="feature-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.page_link(page, label=f"Open →")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.1rem;">'
        'CIFAR-10 Classifier</p>'
        '<p style="font-size:0.75rem;color:#475569;margin-top:0;">Kabir Patil · ResNet CNN · PyTorch</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    val_acc_s = model_meta.get("val_acc", "N/A")
    try:
        val_acc_s = f"{float(val_acc_s):.2f}%"
    except Exception:
        pass
    st.markdown(
        f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:0.8rem 1rem;">'
        f'<div style="font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;">Model Status</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:#60a5fa;margin-top:0.2rem;">90.22%</div>'
        f'<div style="font-size:0.7rem;color:#475569;">test acc · val {val_acc_s} · ep 38</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<p style="font-size:0.75rem;color:#475569;font-weight:600;margin-bottom:0.4rem;">CLASSES</p>'
        '<p style="font-size:0.75rem;color:#64748b;line-height:1.9;">'
        'airplane · automobile · bird<br>cat · deer · dog<br>frog · horse · ship · truck'
        '</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Built with PyTorch + Streamlit")
