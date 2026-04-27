"""
utils/visualization.py
=======================
Reusable Plotly and matplotlib chart helpers for the Streamlit app.
All Plotly charts use plotly.graph_objects for full control.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Consistent colour palette
_BLUE   = "#4F8EF7"
_PURPLE = "#A78BFA"
_GREEN  = "#34D399"
_AMBER  = "#FBBF24"
_RED    = "#F87171"

_TIER_COLORS = lambda acc: _GREEN if acc >= 90 else (_AMBER if acc >= 80 else _RED)


def top_k_bar_chart(
    class_names: List[str],
    confidences: List[float],
    top_k: int = 5,
    title: str = "Top Predictions",
) -> go.Figure:
    """Horizontal Plotly bar chart of top-K class confidences.

    Args:
        class_names:  List of class label strings (length = top_k).
        confidences:  Corresponding confidence scores in ``[0, 1]``.
        top_k:        Number of bars to show.
        title:        Chart title.

    Returns:
        A Plotly Figure object.
    """
    names = class_names[:top_k]
    confs = [c * 100 for c in confidences[:top_k]]
    colors = [_BLUE, _PURPLE, "#60A5FA", "#818CF8", "#C4B5FD"][:top_k]

    fig = go.Figure(go.Bar(
        x=confs, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c:.1f}%" for c in confs],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Confidence (%)", range=[0, 115], gridcolor="#2D3748"),
        yaxis=dict(autorange="reversed", gridcolor="#2D3748"),
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), margin=dict(l=10, r=10, t=40, b=10),
        height=250,
    )
    return fig


def per_class_metrics_chart(
    per_class: Dict[str, Dict[str, float]],
) -> go.Figure:
    """Grouped Plotly bar chart of per-class precision, recall, F1.

    Args:
        per_class: Dict mapping class name → ``{precision, recall, f1}``.

    Returns:
        A Plotly Figure object.
    """
    classes = list(per_class.keys())
    prec    = [per_class[c]["precision"] * 100 for c in classes]
    rec     = [per_class[c]["recall"]    * 100 for c in classes]
    f1      = [per_class[c]["f1"]        * 100 for c in classes]

    fig = go.Figure([
        go.Bar(name="Precision", x=classes, y=prec, marker_color=_BLUE,
               hovertemplate="%{x}<br>Precision: %{y:.1f}%<extra></extra>"),
        go.Bar(name="Recall",    x=classes, y=rec,  marker_color=_PURPLE,
               hovertemplate="%{x}<br>Recall: %{y:.1f}%<extra></extra>"),
        go.Bar(name="F1-Score",  x=classes, y=f1,   marker_color=_GREEN,
               hovertemplate="%{x}<br>F1: %{y:.1f}%<extra></extra>"),
    ])
    fig.update_layout(
        barmode="group", title="Per-Class Metrics",
        xaxis=dict(title="Class", gridcolor="#2D3748"),
        yaxis=dict(title="Score (%)", gridcolor="#2D3748", range=[0, 110]),
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), legend=dict(bgcolor="#1A1D27"),
        height=380,
    )
    return fig


def training_curves_chart(history: Dict[str, List[float]]) -> go.Figure:
    """Dual-y-axis Plotly chart: loss (left) and accuracy (right) vs epoch.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``,
                 ``train_acc``, ``val_acc``.

    Returns:
        A Plotly Figure with secondary y-axis.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = go.Figure()
    # Loss traces (primary y)
    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"], name="Train Loss",
        line=dict(color=_RED, width=2), mode="lines",
        hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"], name="Val Loss",
        line=dict(color=_AMBER, width=2, dash="dash"), mode="lines",
        hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
    ))
    # Accuracy traces (secondary y)
    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_acc"], name="Train Acc",
        line=dict(color=_BLUE, width=2), mode="lines", yaxis="y2",
        hovertemplate="Epoch %{x}<br>Train Acc: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_acc"], name="Val Acc",
        line=dict(color=_GREEN, width=2, dash="dash"), mode="lines", yaxis="y2",
        hovertemplate="Epoch %{x}<br>Val Acc: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Training History",
        xaxis=dict(title="Epoch", gridcolor="#2D3748"),
        yaxis=dict(title="Loss", gridcolor="#2D3748", side="left"),
        yaxis2=dict(title="Accuracy (%)", overlaying="y", side="right",
                    gridcolor="#2D3748", range=[0, 105]),
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), legend=dict(bgcolor="#1A1D27"),
        height=400,
    )
    return fig


def confusion_matrix_fig(
    cm: np.ndarray,
    class_names: List[str] = CIFAR10_CLASSES,
) -> plt.Figure:
    """Normalised seaborn confusion matrix.

    Args:
        cm:           Raw 10×10 confusion matrix (integer counts).
        class_names:  List of class label strings.

    Returns:
        A Matplotlib Figure object for use with ``st.pyplot``.
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1D27")
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", linewidths=0.4, linecolor="#2D3748",
        ax=ax, annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Normalised Confusion Matrix", color="white", fontsize=13, pad=14)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True",      color="white")
    ax.tick_params(colors="white", rotation=45)
    return fig


def per_class_accuracy_bar(cm: np.ndarray) -> go.Figure:
    """Horizontal bar chart of per-class accuracy, sorted descending.

    Args:
        cm: Raw 10×10 confusion matrix.

    Returns:
        Plotly Figure.
    """
    accs  = 100.0 * cm.diagonal() / cm.sum(axis=1)
    order = np.argsort(accs)[::-1]
    names = [CIFAR10_CLASSES[i] for i in order]
    vals  = accs[order]
    colors = [_TIER_COLORS(a) for a in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=vals.mean(), line_dash="dash", line_color="#9CA3AF",
                  annotation_text=f"Mean: {vals.mean():.1f}%", annotation_font_color="#9CA3AF")
    fig.update_layout(
        title="Per-Class Test Accuracy",
        xaxis=dict(title="Accuracy (%)", range=[0, 110], gridcolor="#2D3748"),
        yaxis=dict(autorange="reversed", gridcolor="#2D3748"),
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), height=380,
    )
    return fig


def class_distribution_bar(
    train_counts: Dict[str, int],
    test_counts:  Dict[str, int],
) -> go.Figure:
    """Grouped bar chart showing train vs test counts per class.

    Args:
        train_counts: Dict mapping class name → training sample count.
        test_counts:  Dict mapping class name → test sample count.

    Returns:
        Plotly Figure.
    """
    classes = CIFAR10_CLASSES
    fig = go.Figure([
        go.Bar(name="Train", x=classes, y=[train_counts.get(c, 0) for c in classes],
               marker_color=_BLUE,
               hovertemplate="%{x}<br>Train: %{y:,}<extra></extra>"),
        go.Bar(name="Test",  x=classes, y=[test_counts.get(c, 0)  for c in classes],
               marker_color=_PURPLE,
               hovertemplate="%{x}<br>Test: %{y:,}<extra></extra>"),
    ])
    fig.update_layout(
        barmode="group", title="Class Distribution",
        xaxis=dict(title="Class"),
        yaxis=dict(title="Sample Count", gridcolor="#2D3748"),
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), legend=dict(bgcolor="#1A1D27"),
        height=350,
    )
    return fig


def prediction_pie_chart(class_counts: Dict[str, int]) -> go.Figure:
    """Pie chart of predicted class distribution.

    Args:
        class_counts: Dict mapping class name → count.

    Returns:
        Plotly Figure.
    """
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4,
        marker=dict(colors=[
            "#4F8EF7","#A78BFA","#34D399","#FBBF24","#F87171",
            "#60A5FA","#818CF8","#6EE7B7","#FCD34D","#FCA5A5",
        ]),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} images (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title="Predicted Class Distribution",
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), height=380,
    )
    return fig
