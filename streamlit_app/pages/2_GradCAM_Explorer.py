"""
pages/2_GradCAM_Explorer.py — AUDIT FIXED
FIXES:
- CRITICAL: Removed st.set_page_config() — only app.py may call it.
- CRITICAL: Image loaded via BytesIO from shared_image_bytes (not PIL object).
- CRITICAL: import cv2 moved to top-level — late imports cause silent failures.
- WARNING:  apply_colormap argument order fixed (heatmap first, then img).
- WARNING:  model.eval() explicitly called before GradCAM forward pass.
- WARNING:  GradCAM hooks always removed in finally block (already correct, kept).
- MINOR:    Unused denormalize_tensor import removed.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.gradcam import GradCAM
from utils.preprocessing import CIFAR10_CLASSES, preprocess_image

# ── Guard ─────────────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.error("⚠️ Model not loaded. Please open the **Home** page first.")
    st.stop()

model  = st.session_state["model"]
device = next(model.parameters()).device
model.eval()   # FIX: explicit eval() before any inference

# ── Header ────────────────────────────────────────────────────────────────────
st.title("GradCAM Explorer")
st.markdown(
    "Visualise **where the model is looking** when it makes a prediction. "
    "GradCAM highlights the spatial regions that contributed most to the decision."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ GradCAM Controls")
    layer_choice = st.radio(
        "Target Layer",
        options=["Block 1 (64ch)", "Block 2 (128ch)", "Block 3 (256ch)"],
        index=2,
        help="Block 3 gives the most semantically rich heatmap.",
    )
    colormap = st.selectbox("Heatmap Colormap", ["jet","viridis","plasma","inferno"], index=0)
    alpha    = st.slider("Overlay Opacity", 0.2, 0.9, 0.5, 0.05)
    st.divider()
    st.markdown("**Tip:** Block 3 → high-level features; Block 1 → edges/textures.")

layer_map = {
    "Block 1 (64ch)":  model.block1,
    "Block 2 (128ch)": model.block2,
    "Block 3 (256ch)": model.block3,
}
target_layer = layer_map[layer_choice]

# ── Image source ──────────────────────────────────────────────────────────────
st.subheader("📤 Image Source")
src_tab, carry_tab = st.tabs(["Upload New", "Use Image from Prediction Page"])

pil_image: Image.Image | None = None

with src_tab:
    f = st.file_uploader(
        "Upload image", type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed", key="gradcam_uploader",
    )
    if f is not None:
        try:
            f.seek(0)
            pil_image = Image.open(io.BytesIO(f.read())).convert("RGB")
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            st.session_state["shared_image_bytes"] = buf.getvalue()
        except Exception as e:
            st.error(f"❌ Could not open image: {e}")

with carry_tab:
    # FIX: read from bytes in session_state, not PIL object
    if "shared_image_bytes" in st.session_state:
        pil_image = Image.open(
            io.BytesIO(st.session_state["shared_image_bytes"])
        ).convert("RGB")
        st.success("✅ Using image carried from the Live Prediction page.", icon="✅")
        st.image(pil_image.resize((96, 96)), caption="Carried image")
    else:
        st.info("No image in session yet. Upload one on the Live Prediction page first.", icon="ℹ️")

# ── GradCAM computation ───────────────────────────────────────────────────────
if pil_image is None:
    st.info("👈 Upload an image to generate the GradCAM heatmap.", icon="ℹ️")
else:
    tensor = preprocess_image(pil_image)           # (1, 3, 32, 32)

    with st.spinner("Computing GradCAM…"):
        gradcam = GradCAM(model, target_layer)
        try:
            heatmap = gradcam.generate(tensor, device=device)   # (h, w) float32
        except Exception as exc:
            st.error(f"GradCAM failed: {exc}")
            st.stop()
        finally:
            gradcam.remove_hooks()

    # Prepare display-size original (224×224 for clarity)
    orig_large = np.array(pil_image.convert("RGB").resize((224, 224)))

    # Resize heatmap to display size
    heatmap_big = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)

    # FIX: correct argument order — apply_colormap(heatmap, original_img, ...)
    orig_disp, heatmap_rgb, overlay = GradCAM.apply_colormap(
        heatmap_big, orig_large, alpha=alpha, colormap_name=colormap
    )

    # Quick inference for label (separate no_grad block — not inside GradCAM)
    with torch.no_grad():
        logits    = model(tensor.to(device))
        probs     = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx  = int(np.argmax(probs))
        pred_cls  = CIFAR10_CLASSES[pred_idx]
        pred_conf = float(probs[pred_idx])

    st.markdown(
        f"**Predicted class:** `{pred_cls}` &nbsp;|&nbsp; "
        f"**Confidence:** `{pred_conf:.1%}` &nbsp;|&nbsp; "
        f"**Target layer:** `{layer_choice}`"
    )
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.image(orig_disp,   caption="📷 Original Image",       use_container_width=True, clamp=True)
    c2.image(heatmap_rgb, caption=f"🌡️ GradCAM ({colormap})",use_container_width=True, clamp=True)
    c3.image(overlay,     caption="🔥 Overlay (blended)",    use_container_width=True, clamp=True)

    st.caption(
        f"Red/bright = high importance for **{pred_cls}** via **{layer_choice}**."
    )

    with st.expander("ℹ️ How does GradCAM work?"):
        st.markdown("""
**GradCAM (Gradient-weighted Class Activation Mapping)** produces visual explanations without model surgery.

**Step 1 — Forward hook:** captures feature maps (activations) from the target layer during the forward pass.

**Step 2 — Gradient hook:** during backprop from the target class score, captures gradients w.r.t. those activations. Global average pooling gives one importance weight per channel: α_k = GAP(∂Y^c / ∂A_k).

**Step 3 — Weighted sum + ReLU:** `CAM = ReLU(Σ_k α_k · A_k)`. ReLU keeps only positively-influential regions. The result is upsampled to input resolution and blended with the original.
        """)

    with st.expander("📊 Heatmap Statistics"):
        s1, s2, s3 = st.columns(3)
        s1.metric("Max Activation",     f"{float(heatmap.max()):.3f}")
        s2.metric("Mean Activation",    f"{float(heatmap.mean()):.3f}")
        s3.metric("High-attention area",f"{float((heatmap > 0.5).mean()):.1%}")
