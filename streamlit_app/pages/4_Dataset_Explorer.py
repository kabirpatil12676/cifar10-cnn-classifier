"""
pages/4_Dataset_Explorer.py — AUDIT FIXED
FIXES:
- CRITICAL: Removed st.set_page_config() — only app.py may call it.
- CRITICAL: /tmp/cifar10 is Linux-only; use pathlib for cross-platform cache dir.
- CRITICAL: Iterating the full dataset (50K items) is extremely slow — replaced
            with a vectorised index lookup using dataset.targets list.
- WARNING:  Image.NEAREST deprecated in Pillow 10 → Image.Resampling.NEAREST.
- WARNING:  Both if/else branches for class distribution had identical code —
            simplified to a single assignment.
- WARNING:  plotly.graph_objects imported at module level (not mid-file).
- WARNING:  torch imported but never actually used — removed.
- MINOR:    aug_names variable assigned but never used — removed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocessing import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, get_augmentation_samples
from utils.visualization  import class_distribution_bar

# FIX: Pillow 10+ compat
_NEAREST = Image.Resampling.NEAREST

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Dataset Explorer")
st.markdown(
    "Explore the **CIFAR-10** dataset: 60,000 32×32 colour images across 10 classes."
)

# ── Sidebar controls (defined BEFORE use) ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    selected_classes = st.multiselect(
        "Show classes", CIFAR10_CLASSES, default=CIFAR10_CLASSES[:5],
    )
    n_cols = st.slider("Samples per class", 4, 10, 6)

# ── Load dataset ──────────────────────────────────────────────────────────────
# FIX: use a cross-platform temp dir instead of hardcoded /tmp/cifar10
_CACHE_DIR = Path(tempfile.gettempdir()) / "cifar10_streamlit"

@st.cache_data(show_spinner="Loading CIFAR-10 (first run ~170 MB download)…")
def load_cifar10():
    """Download CIFAR-10 and return (train_ds, test_ds)."""
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    tfm      = transforms.ToTensor()
    train_ds = CIFAR10(root=str(_CACHE_DIR), train=True,  download=True, transform=tfm)
    test_ds  = CIFAR10(root=str(_CACHE_DIR), train=False, download=True, transform=tfm)
    return train_ds, test_ds

dataset_loaded = False
train_ds = test_ds = None
try:
    train_ds, test_ds = load_cifar10()
    dataset_loaded = True
except Exception as exc:
    st.warning(f"Could not load CIFAR-10: {exc}. Showing placeholder data.", icon="⚠️")

# ── 1. Class Distribution ─────────────────────────────────────────────────────
st.subheader("📊 Class Distribution")
# FIX: both branches were identical — simplified
train_counts = {cls: 5000 for cls in CIFAR10_CLASSES}
test_counts  = {cls: 1000 for cls in CIFAR10_CLASSES}
st.plotly_chart(class_distribution_bar(train_counts, test_counts), width='stretch')
st.caption("CIFAR-10 is **perfectly balanced** — 5,000 train + 1,000 test images per class.")

st.divider()

# ── 2. Sample Image Grid ──────────────────────────────────────────────────────
st.subheader("🖼️ Sample Image Grid")

if dataset_loaded and selected_classes:
    # FIX: vectorised lookup instead of iterating 50K samples sequentially
    targets = dataset_loaded and hasattr(train_ds, "targets") and train_ds.targets

    class_samples: dict[str, list[Image.Image]] = {c: [] for c in selected_classes}

    if targets is not False:
        # targets is a list of ints — build index per class first
        class_indices: dict[str, list[int]] = {c: [] for c in selected_classes}
        for idx, lbl in enumerate(train_ds.targets):
            cls_name = CIFAR10_CLASSES[lbl]
            if cls_name in class_indices:
                class_indices[cls_name].append(idx)

        for cls_name in selected_classes:
            for img_idx in class_indices[cls_name][:n_cols]:
                img_t, _ = train_ds[img_idx]
                pil = Image.fromarray(
                    (img_t.permute(1,2,0).numpy() * 255).astype(np.uint8)
                )
                class_samples[cls_name].append(pil.resize((64,64), _NEAREST))

    for cls_name in selected_classes:
        st.markdown(f"**{cls_name.capitalize()}**")
        img_cols = st.columns(n_cols)
        for col, img in zip(img_cols, class_samples.get(cls_name, [])):
            col.image(img, width='stretch')

elif selected_classes:
    st.info("Dataset not loaded — showing placeholder tiles.", icon="ℹ️")
    rng = np.random.default_rng(0)
    for cls_name in selected_classes:
        st.markdown(f"**{cls_name.capitalize()}** (placeholder)")
        img_cols = st.columns(n_cols)
        for col in img_cols:
            rgb = tuple(rng.integers(80, 200, 3).tolist())
            col.image(Image.new("RGB", (64,64), color=rgb), width='stretch')
else:
    st.info("Select at least one class in the sidebar.", icon="ℹ️")

st.divider()

# ── 3. Pixel Statistics ───────────────────────────────────────────────────────
st.subheader("📈 Channel-wise Pixel Statistics")

channels = ["Red (R)", "Green (G)", "Blue (B)"]
fig_stats = go.Figure([
    go.Bar(name="Mean", x=channels, y=list(CIFAR10_MEAN),
           marker_color="#4F8EF7",
           text=[f"{v:.4f}" for v in CIFAR10_MEAN], textposition="outside"),
    go.Bar(name="Std",  x=channels, y=list(CIFAR10_STD),
           marker_color="#A78BFA",
           text=[f"{v:.4f}" for v in CIFAR10_STD],  textposition="outside"),
])
fig_stats.update_layout(
    barmode="group", title="CIFAR-10 Channel-wise Mean & Std (pre-computed)",
    yaxis=dict(title="Value", gridcolor="#2D3748", range=[0, 0.7]),
    plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
    font=dict(color="#FAFAFA"), height=320,
)
st.plotly_chart(fig_stats, use_container_width=True)

st.divider()

# ── 4. Augmentation Preview ───────────────────────────────────────────────────
st.subheader("🎨 Data Augmentation Preview")
st.markdown("See how training augmentations transform the same image.")

if dataset_loaded and train_ds is not None:
    sample_cls = st.selectbox("Pick a class for augmentation demo", CIFAR10_CLASSES, index=0)
    # Vectorised lookup
    idx0 = next(
        (i for i, lbl in enumerate(train_ds.targets) if CIFAR10_CLASSES[lbl] == sample_cls),
        0,
    )
    raw_t, _ = train_ds[idx0]
    raw_pil   = Image.fromarray((raw_t.permute(1,2,0).numpy() * 255).astype(np.uint8))
else:
    raw_pil = Image.new("RGB", (32, 32), color=(100, 150, 200))

aug_samples = get_augmentation_samples(raw_pil)
aug_cols    = st.columns(3)
for idx, (name, aug_img) in enumerate(aug_samples.items()):
    aug_cols[idx % 3].image(aug_img, caption=name, width='stretch')

st.divider()

# ── 5. Dataset Stats Table ────────────────────────────────────────────────────
st.subheader("📋 Dataset Statistics")
df_stats = pd.DataFrame([
    {"Class": cls, "Train": 5000, "Test": 1000, "Size": "32×32", "Channels": 3}
    for cls in CIFAR10_CLASSES
])
st.dataframe(df_stats, use_container_width=True, hide_index=True)
st.caption(
    f"Total: {len(CIFAR10_CLASSES)*6000:,} images | "
    f"Mean: {CIFAR10_MEAN} | Std: {CIFAR10_STD}"
)
