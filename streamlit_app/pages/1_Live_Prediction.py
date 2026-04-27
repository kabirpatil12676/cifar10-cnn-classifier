"""
pages/1_Live_Prediction.py — AUDIT FIXED
FIXES:
- CRITICAL: Removed st.set_page_config() — only app.py may call it.
- CRITICAL: UploadedFile read with BytesIO wrapper (file may already be read).
- WARNING:  PIL import moved to top-level (not inside conditionals).
- WARNING:  pil_image initialised outside tab blocks to avoid scope leakage.
- WARNING:  session_state["shared_image"] stores bytes, not PIL object,
            to avoid storing large objects in session_state.
- MINOR:    _make_placeholder_image uses hex-to-RGB conversion properly.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocessing import (
    CIFAR10_CLASSES, preprocess_image, get_augmentation_samples,
)
from utils.visualization import top_k_bar_chart

# ── Guard: model must be loaded via app.py first ──────────────────────────────
if "model" not in st.session_state:
    st.error("⚠️ Model not loaded. Please open the **Home** page first to initialise the model.")
    st.stop()

model      = st.session_state["model"]
model_meta = st.session_state.get("model_meta", {})
device     = next(model.parameters()).device

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🖼️ Live Prediction")
st.markdown("Upload any image and get an instant CIFAR-10 classification with confidence scores.")

if model_meta.get("is_fallback"):
    st.warning("⚠️ Using randomly-initialized weights — predictions are not meaningful.", icon="⚠️")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    conf_threshold  = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
                                 help="Flag predictions below this threshold.")
    top_k           = st.radio("Show Top-K", options=[3, 5], index=1, horizontal=True)
    show_preprocess = st.toggle("Show preprocessing preview", value=False)
    st.divider()
    st.caption("Image is resized to 32×32 and normalised using CIFAR-10 channel statistics.")

# ── Placeholder image generator ───────────────────────────────────────────────
_CLASS_COLORS = {
    "airplane":"#3B82F6","automobile":"#EF4444","bird":"#10B981",
    "cat":"#F59E0B","deer":"#6366F1","dog":"#EC4899",
    "frog":"#14B8A6","horse":"#8B5CF6","ship":"#0EA5E9","truck":"#F97316",
}

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

def _make_placeholder(cls: str) -> Image.Image:
    rgb = _hex_to_rgb(_CLASS_COLORS.get(cls, "#4F8EF7"))
    img  = Image.new("RGB", (128, 128), color=rgb)
    draw = ImageDraw.Draw(img)
    draw.text((10, 54), cls.upper(), fill="white")
    return img

# ── Layout ────────────────────────────────────────────────────────────────────
col_up, col_res = st.columns([1, 1], gap="large")

pil_image: Image.Image | None = None

with col_up:
    st.subheader("📤 Input")
    tab_upload, tab_sample = st.tabs(["Upload Image", "Try a Sample"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg","jpeg","png","bmp","webp"],
            label_visibility="collapsed",
            key="live_uploader",
        )
        if uploaded_file is not None:
            try:
                # FIX: wrap with BytesIO so repeated reads work correctly
                uploaded_file.seek(0)
                pil_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
                # Store as bytes in session_state (not PIL object — avoids bloat)
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                st.session_state["shared_image_bytes"] = buf.getvalue()
            except Exception as e:
                st.error(f"❌ Could not open image: {e}")

    with tab_sample:
        sel_class = st.selectbox("Pick a CIFAR-10 class", CIFAR10_CLASSES, key="sample_class")
        if st.button("Use this sample", use_container_width=True):
            pil_image = _make_placeholder(sel_class)
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            st.session_state["shared_image_bytes"] = buf.getvalue()

    # Restore from session if no new upload this run
    if pil_image is None and "shared_image_bytes" in st.session_state:
        pil_image = Image.open(io.BytesIO(st.session_state["shared_image_bytes"])).convert("RGB")

    if pil_image is not None:
        st.image(pil_image.resize((200, 200)), caption="Input image", use_container_width=False)

    if show_preprocess and pil_image is not None:
        with st.expander("🔬 Preprocessing Preview", expanded=True):
            samples  = get_augmentation_samples(pil_image)
            aug_cols = st.columns(3)
            for idx, (name, aug_img) in enumerate(samples.items()):
                aug_cols[idx % 3].image(aug_img, caption=name, use_container_width=True)

# ── Inference ─────────────────────────────────────────────────────────────────
with col_res:
    st.subheader("📊 Results")

    if pil_image is None:
        st.info("👈 Upload an image or select a sample to see predictions.", icon="ℹ️")
    else:
        tensor = preprocess_image(pil_image)   # (1, 3, 32, 32)

        t_start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor.to(device))
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        top_indices = np.argsort(probs)[::-1]
        top_classes = [CIFAR10_CLASSES[i] for i in top_indices[:top_k]]
        top_confs   = [float(probs[i])     for i in top_indices[:top_k]]

        pred_class = top_classes[0]
        pred_conf  = top_confs[0]

        if pred_conf >= conf_threshold:
            st.success(f"**{pred_class.upper()}**  —  {pred_conf:.1%} confidence", icon="✅")
        else:
            st.error(
                f"**{pred_class.upper()}**  —  {pred_conf:.1%}  "
                f"⚠️ Below threshold ({conf_threshold:.0%})",
                icon="⚠️",
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("🏆 Prediction", pred_class.capitalize())
        m2.metric("📊 Confidence", f"{pred_conf:.1%}")
        m3.metric("⚡ Inference",  f"{elapsed_ms:.1f} ms")

        # FIX: st.progress requires float in [0.0, 1.0]
        st.progress(float(pred_conf), text=f"Top-1 confidence: {pred_conf:.1%}")

        fig = top_k_bar_chart(top_classes, top_confs, top_k=top_k,
                              title=f"Top-{top_k} Predictions")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"🔍 What does a '{pred_class}' look like?"):
            st.markdown(
                f"Placeholder tiles for **{pred_class}**. "
                "In production these would be real CIFAR-10 test images."
            )
            exp_cols = st.columns(4)
            for i, ec in enumerate(exp_cols):
                ec.image(_make_placeholder(pred_class),
                         caption=f"{pred_class} #{i+1}", use_container_width=True)
