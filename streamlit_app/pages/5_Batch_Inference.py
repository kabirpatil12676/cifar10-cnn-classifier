"""
pages/5_Batch_Inference.py — AUDIT FIXED
FIXES:
- CRITICAL: Removed st.set_page_config() — only app.py may call it.
- CRITICAL: Each UploadedFile.read() wrapped in BytesIO + seek(0) so re-reads
            don't return empty bytes.
- CRITICAL: CSV download uses .encode('utf-8') → bytes, not StringIO.getvalue()
            string (st.download_button data= should be bytes on cloud).
- WARNING:  dataframe style.apply with axis=0 applies per column, not per row.
            Fixed to use axis=1 (per row) for the row-level highlight.
- WARNING:  try/except around each Image.open() so one bad file doesn't crash
            the entire batch.
- WARNING:  Image.LANCZOS deprecated alias — use Image.Resampling.LANCZOS.
- MINOR:    progress bar always receives float (was already correct, kept).
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocessing import CIFAR10_CLASSES, preprocess_image
from utils.visualization  import prediction_pie_chart

# FIX: Pillow 10+ compat
_LANCZOS = Image.Resampling.LANCZOS

# ── Guard ─────────────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.error("⚠️ Model not loaded. Please open the **Home** page first.")
    st.stop()

model  = st.session_state["model"]
device = next(model.parameters()).device
model.eval()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Batch Inference")
st.markdown(
    "Upload up to **50 images** and run inference on all of them at once. "
    "Download the results as CSV."
)

if st.session_state.get("model_meta", {}).get("is_fallback"):
    st.warning("⚠️ Using randomly-initialized weights — predictions are not meaningful.", icon="⚠️")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    show_thumbnails = st.toggle("Show image thumbnails", value=True)
    conf_threshold  = st.slider("Flag low-confidence below", 0.0, 1.0, 0.5, 0.05)

# ── Uploader ──────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload images (jpg, jpeg, png, bmp, webp) — max 50",
    type=["jpg","jpeg","png","bmp","webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key="batch_uploader",
)

if not uploaded_files:
    st.info("👆 Upload one or more images to start batch inference.", icon="ℹ️")
    st.stop()

if len(uploaded_files) > 50:
    st.warning(
        f"⚠️ {len(uploaded_files)} files uploaded — only the first 50 will be processed.",
        icon="⚠️",
    )
    uploaded_files = uploaded_files[:50]

# ── Batch inference ───────────────────────────────────────────────────────────
st.subheader(f"⚙️ Processing {len(uploaded_files)} image(s)…")
progress_bar = st.progress(0.0, text="Starting…")

results:    list[dict] = []
thumbnails: list[Image.Image] = []

for idx, uf in enumerate(uploaded_files):
    try:
        # FIX: seek(0) + BytesIO wrapper for reliable multi-read
        uf.seek(0)
        pil_img = Image.open(io.BytesIO(uf.read())).convert("RGB")
    except Exception as exc:
        st.warning(f"⚠️ Skipping `{uf.name}`: could not open image ({exc})")
        progress_bar.progress((idx + 1) / len(uploaded_files),
                              text=f"Skipped {uf.name}")
        continue

    # FIX: Pillow 10+ LANCZOS
    thumb  = pil_img.resize((48, 48), _LANCZOS)
    tensor = preprocess_image(pil_img)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    inf_ms = (time.perf_counter() - t0) * 1000

    sorted_idx = np.argsort(probs)[::-1]
    pred1_cls  = CIFAR10_CLASSES[sorted_idx[0]]
    pred1_conf = float(probs[sorted_idx[0]])
    pred2_cls  = CIFAR10_CLASSES[sorted_idx[1]]
    pred2_conf = float(probs[sorted_idx[1]])

    results.append({
        "filename":          uf.name,
        "predicted_class":   pred1_cls,
        "confidence":        round(pred1_conf, 4),
        "confidence_%":      f"{pred1_conf:.1%}",
        "top2_class":        pred2_cls,
        "top2_confidence":   round(pred2_conf, 4),
        "inference_time_ms": round(inf_ms, 2),
        "low_confidence":    pred1_conf < conf_threshold,
    })
    thumbnails.append(thumb)

    # FIX: always float for st.progress
    progress_bar.progress(
        float((idx + 1) / len(uploaded_files)),
        text=f"Processing {idx+1}/{len(uploaded_files)}: {uf.name}",
    )

if not results:
    st.error("No images could be processed. Please check your uploaded files.")
    st.stop()

progress_bar.progress(1.0, text=f"✅ Done! Processed {len(results)} image(s).")

# ── Results ───────────────────────────────────────────────────────────────────
st.subheader("📊 Results")
df = pd.DataFrame(results)

n_low = int(df["low_confidence"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("📷 Processed",       len(df))
c2.metric("🏆 Most Predicted",  df["predicted_class"].value_counts().idxmax())
c3.metric("⚠️ Low Confidence",  n_low, help=f"Below {conf_threshold:.0%} threshold")
c4.metric("⚡ Avg Inference",   f"{df['inference_time_ms'].mean():.1f} ms")

if show_thumbnails:
    st.markdown("**Predictions (with thumbnails):**")
    for row_idx, (_, row) in enumerate(df.iterrows()):
        img_col, data_col = st.columns([1, 8])
        img_col.image(thumbnails[row_idx], use_container_width=True)
        flag = "⚠️ Low" if row["low_confidence"] else "✅"
        data_col.markdown(
            f"**{row['filename']}** → **{row['predicted_class']}** "
            f"({row['confidence_%']}) {flag} &nbsp;|&nbsp; "
            f"2nd: {row['top2_class']} ({row['top2_confidence']:.1%}) &nbsp;|&nbsp; "
            f"⏱ {row['inference_time_ms']} ms"
        )
else:
    # FIX: axis=1 for per-row styling
    display_df = df.drop(columns=["low_confidence", "confidence_%"])
    def _row_style(row):
        flag = df.loc[row.name, "low_confidence"]
        return ["background-color: #3d1515" if flag else ""] * len(row)
    st.dataframe(
        display_df.style.apply(_row_style, axis=1),
        use_container_width=True, hide_index=True,
    )

st.divider()

st.subheader("🥧 Predicted Class Distribution")
class_counts = df["predicted_class"].value_counts().to_dict()
st.plotly_chart(prediction_pie_chart(class_counts), use_container_width=True)

st.divider()

# ── CSV download ──────────────────────────────────────────────────────────────
st.subheader("💾 Download Results")
csv_df  = df.drop(columns=["low_confidence", "confidence_%"])
# FIX: encode to bytes for reliable download on Streamlit Cloud
csv_bytes = csv_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download CSV",
    data=csv_bytes,           # FIX: bytes, not string
    file_name="cifar10_batch_predictions.csv",
    mime="text/csv",
    use_container_width=True,
)
st.caption("CSV columns: filename, predicted_class, confidence, top2_class, top2_confidence, inference_time_ms")
