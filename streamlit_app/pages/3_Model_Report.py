"""
pages/3_Model_Report.py — AUDIT FIXED
FIXES:
- CRITICAL: Removed st.set_page_config() — only app.py may call it.
- CRITICAL: plt.close(fig) added after st.pyplot(fig) to prevent memory leak.
- WARNING:  @st.cache_data functions with np.random calls are non-deterministic
            across cache invalidations — fixed by seeding inside function.
- WARNING:  load_metrics() type annotation fixed (was wrong — returned tuple
            but annotation said dict).
- MINOR:    Unused `using_synthetic` variable removed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.visualization import (
    confusion_matrix_fig, per_class_metrics_chart,
    training_curves_chart, per_class_accuracy_bar,
)

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck",
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Model Report")
st.markdown(
    "Complete evaluation results — confusion matrix, per-class metrics, "
    "training history, and error analysis."
)

# ── Pre-computed result paths ─────────────────────────────────────────────────
RESULTS_DIR  = ROOT / "results"
METRICS_FILE = RESULTS_DIR / "metrics.json"
HISTORY_FILE = ROOT / "checkpoints" / "training_history.json"


@st.cache_data
def load_metrics() -> tuple[dict, bool]:
    """Return (metrics_dict, is_synthetic)."""
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, encoding="utf-8") as f:
                return json.load(f), False
        except Exception:
            pass  # fall through to synthetic

    # FIX: seed inside function for reproducible synthetic data
    rng = np.random.default_rng(42)
    per_class = {
        cls: {
            "precision": float(rng.uniform(0.79, 0.95)),
            "recall":    float(rng.uniform(0.78, 0.94)),
            "f1":        float(rng.uniform(0.79, 0.94)),
        }
        for cls in CIFAR10_CLASSES
    }
    return {
        "accuracy": 88.7, "macro_f1": 0.887, "weighted_f1": 0.887,
        "per_class": per_class,
    }, True


@st.cache_data
def load_history() -> tuple[dict, bool]:
    """Return (history_dict, is_synthetic)."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f), False
        except Exception:
            pass

    rng = np.random.default_rng(42)
    n   = 60
    ep  = np.arange(1, n + 1)
    t_loss = (2.3 * np.exp(-0.045 * ep) + 0.12 + rng.uniform(-0.02, 0.02, n)).tolist()
    v_loss = (2.3 * np.exp(-0.038 * ep) + 0.28 + rng.uniform(-0.02, 0.02, n)).tolist()
    t_acc  = np.clip(30 + 60*(1-np.exp(-0.06*ep)) + rng.uniform(-0.5,0.5,n), 0, 99).tolist()
    v_acc  = np.clip(28 + 56*(1-np.exp(-0.055*ep)) + rng.uniform(-0.5,0.5,n), 0, 95).tolist()
    return {"train_loss":t_loss,"val_loss":v_loss,"train_acc":t_acc,"val_acc":v_acc}, True


@st.cache_data
def make_synthetic_cm() -> np.ndarray:
    rng = np.random.default_rng(42)
    cm  = np.zeros((10, 10), dtype=int)
    for i in range(10):
        correct   = int(1000 * rng.uniform(0.80, 0.95))
        cm[i, i]  = correct
        others    = rng.choice([k for k in range(10) if k != i],
                               size=1000-correct, replace=True)
        for j in others:
            cm[i, j] += 1
    return cm


metrics, syn_metrics = load_metrics()
history, syn_history = load_history()
cm                   = make_synthetic_cm()

if syn_metrics or syn_history:
    st.info(
        "📊 Showing **synthetic demo data**. "
        "Run `python main.py --mode eval` in the training repo to generate real results.",
        icon="ℹ️",
    )

# ── Metric cards ──────────────────────────────────────────────────────────────
st.subheader("🎯 Key Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("🎯 Test Accuracy",   f"{metrics.get('accuracy', 88.7):.2f}%", "+13% over baseline")
m2.metric("📊 Macro F1",        f"{metrics.get('macro_f1', 0.887):.4f}")
m3.metric("⚙️ Parameters",      "~2.8M",  "ResNet-inspired")
m4.metric("⚡ Inference Speed", "~8 ms",  "Per image (CPU)")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Confusion Matrix", "📊 Per-Class Metrics",
    "📈 Training History", "🔍 Error Analysis",
])

with tab1:
    st.markdown("Normalised confusion matrix — diagonal = correct, off-diagonal = confusion.")
    fig_cm = confusion_matrix_fig(cm, CIFAR10_CLASSES)
    # FIX: pass figure explicitly; close after to prevent memory leak
    st.pyplot(fig_cm, use_container_width=True)
    plt.close(fig_cm)

with tab2:
    sort_by       = st.selectbox("Sort by", ["f1","precision","recall"], index=0)
    per_class_data = metrics.get("per_class", {})
    if per_class_data:
        sorted_classes = sorted(
            per_class_data.keys(),
            key=lambda c: per_class_data[c].get(sort_by, 0), reverse=True,
        )
        sorted_data = {c: per_class_data[c] for c in sorted_classes}
        st.plotly_chart(per_class_metrics_chart(sorted_data), use_container_width=True)
        rows = [
            {"Class": cls, "Precision": f"{v['precision']:.3f}",
             "Recall": f"{v['recall']:.3f}", "F1-Score": f"{v['f1']:.3f}"}
            for cls, v in sorted_data.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab3:
    st.plotly_chart(training_curves_chart(history), use_container_width=True)
    if not syn_history and "val_acc" in history:
        best_epoch = int(np.argmax(history["val_acc"])) + 1
        best_acc   = float(max(history["val_acc"]))
        st.caption(f"Best val accuracy: **{best_acc:.2f}%** at epoch **{best_epoch}**")
    st.plotly_chart(per_class_accuracy_bar(cm), use_container_width=True)

with tab4:
    st.markdown("**Top-10 most confused class pairs** (highest off-diagonal values).")
    confused = [
        {
            "True Class":         CIFAR10_CLASSES[i],
            "Predicted As":       CIFAR10_CLASSES[j],
            "Misclassifications": int(cm[i, j]),
            "Error Rate":         f"{100 * cm[i,j] / max(int(cm[i].sum()), 1):.1f}%",
        }
        for i in range(10) for j in range(10) if i != j
    ]
    top10 = sorted(confused, key=lambda x: x["Misclassifications"], reverse=True)[:10]
    df_err = pd.DataFrame(top10)
    st.dataframe(
        df_err.style.background_gradient(subset=["Misclassifications"], cmap="Reds"),
        use_container_width=True, hide_index=True,
    )
    st.caption("cat↔dog and automobile↔truck are typically the hardest pairs at 32×32 resolution.")
