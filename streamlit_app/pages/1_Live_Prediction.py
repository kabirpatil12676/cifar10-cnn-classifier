"""
pages/1_Live_Prediction.py  —  Real-time CIFAR-10 inference.
"""
from __future__ import annotations

import io
import sys
import tempfile
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

_CIFAR_CACHE = Path(tempfile.gettempdir()) / "cifar10_streamlit"


@st.cache_data(show_spinner=False)
def _get_class_samples(class_name: str, n: int = 4) -> list[bytes]:
    """Return n real CIFAR-10 test images for class_name as PNG bytes."""
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        ds  = CIFAR10(root=str(_CIFAR_CACHE), train=False, download=True,
                      transform=transforms.ToTensor())
        idx = CIFAR10_CLASSES.index(class_name)
        results: list[bytes] = []
        for img_t, lbl in ds:
            if lbl == idx:
                pil = Image.fromarray(
                    (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                ).resize((96, 96), Image.Resampling.NEAREST)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                results.append(buf.getvalue())
                if len(results) == n:
                    break
        return results
    except Exception:
        return []


# ── Guard ─────────────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.error("Model not loaded. Please open the Home page first.")
    st.stop()

model      = st.session_state["model"]
model_meta = st.session_state.get("model_meta", {})
device     = next(model.parameters()).device

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Live Prediction")
st.caption("Upload any image — the model classifies it into one of 10 CIFAR-10 categories.")

if model_meta.get("is_fallback"):
    st.warning("Using randomly-initialized weights. Predictions are not meaningful.", icon="⚠️")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### CIFAR-10 Classifier")
    st.caption("Kabir Patil · ResNet CNN · PyTorch")
    st.divider()
    conf_threshold  = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    top_k           = st.radio("Top-K Predictions", options=[3, 5], index=1, horizontal=True)
    show_preprocess = st.toggle("Show augmentation preview", value=False)
    st.divider()
    st.caption("Image resized to 32x32, normalised to CIFAR-10 channel stats.")

# ── Colour map for placeholder fallback ───────────────────────────────────────
_CLASS_COLORS = {
    "airplane":"#3B82F6","automobile":"#EF4444","bird":"#10B981",
    "cat":"#F59E0B","deer":"#6366F1","dog":"#EC4899",
    "frog":"#14B8A6","horse":"#8B5CF6","ship":"#0EA5E9","truck":"#F97316",
}

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))   # type: ignore

def _make_placeholder(cls: str) -> Image.Image:
    rgb  = _hex_to_rgb(_CLASS_COLORS.get(cls, "#4F8EF7"))
    img  = Image.new("RGB", (96, 96), color=rgb)
    draw = ImageDraw.Draw(img)
    draw.text((6, 40), cls.upper(), fill="white")
    return img

# ── Layout ────────────────────────────────────────────────────────────────────
col_up, col_res = st.columns([1, 1], gap="large")

pil_image: Image.Image | None = None

with col_up:
    st.subheader("Input Image")
    tab_upload, tab_sample = st.tabs(["Upload", "Sample"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg","jpeg","png","bmp","webp"],
            label_visibility="collapsed", key="live_uploader",
        )
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                pil_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                st.session_state["shared_image_bytes"] = buf.getvalue()
            except Exception as e:
                st.error(f"Could not open image: {e}")

    with tab_sample:
        sel_class = st.selectbox("Select class", CIFAR10_CLASSES, key="sample_class")
        if st.button("Load sample", use_container_width=True):
            samples = _get_class_samples(sel_class, n=1)
            if samples:
                pil_image = Image.open(io.BytesIO(samples[0])).convert("RGB")
            else:
                pil_image = _make_placeholder(sel_class)
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            st.session_state["shared_image_bytes"] = buf.getvalue()

    if pil_image is None and "shared_image_bytes" in st.session_state:
        pil_image = Image.open(
            io.BytesIO(st.session_state["shared_image_bytes"])
        ).convert("RGB")

    if pil_image is not None:
        st.image(pil_image.resize((200, 200)), caption="Input image",
                 use_container_width=False)

    if show_preprocess and pil_image is not None:
        with st.expander("Augmentation Preview", expanded=True):
            samples_aug = get_augmentation_samples(pil_image)
            aug_cols    = st.columns(3)
            for idx, (name, aug_img) in enumerate(samples_aug.items()):
                aug_cols[idx % 3].image(aug_img, caption=name, use_container_width=True)

# ── Inference ─────────────────────────────────────────────────────────────────
with col_res:
    st.subheader("Prediction")

    if pil_image is None:
        st.info("Upload an image or load a sample to see predictions.", icon="ℹ️")
    else:
        tensor = preprocess_image(pil_image)

        t_start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor.to(device))
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        top_indices = np.argsort(probs)[::-1]
        top_classes = [CIFAR10_CLASSES[i] for i in top_indices[:top_k]]
        top_confs   = [float(probs[i])    for i in top_indices[:top_k]]

        pred_class = top_classes[0]
        pred_conf  = top_confs[0]

        if pred_conf >= conf_threshold:
            st.success(f"**{pred_class.upper()}** — {pred_conf:.1%} confidence", icon="✅")
        else:
            st.error(
                f"**{pred_class.upper()}** — {pred_conf:.1%}  "
                f"(below threshold {conf_threshold:.0%})", icon="⚠️",
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction",  pred_class.capitalize())
        m2.metric("Confidence",  f"{pred_conf:.1%}")
        m3.metric("Inference",   f"{elapsed_ms:.1f} ms")

        st.progress(float(pred_conf), text=f"Top-1 confidence: {pred_conf:.1%}")

        fig = top_k_bar_chart(top_classes, top_confs, top_k=top_k,
                              title=f"Top-{top_k} Predictions")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"Real CIFAR-10 examples — {pred_class}"):
            real_imgs = _get_class_samples(pred_class, n=4)
            if real_imgs:
                exp_cols = st.columns(4)
                for i, (ec, img_bytes) in enumerate(zip(exp_cols, real_imgs)):
                    ec.image(img_bytes, caption=f"{pred_class} #{i+1}",
                             use_container_width=True)
            else:
                st.info("CIFAR-10 samples not available.", icon="ℹ️")
