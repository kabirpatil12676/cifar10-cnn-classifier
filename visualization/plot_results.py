"""
visualization/plot_results.py
==============================
All publication-quality plots for the CIFAR-10 project.

Plots generated:
    1. sample_grid         — 5×10 grid of CIFAR-10 class samples
    2. training_curves     — loss & accuracy curves (train vs val)
    3. confusion_matrix    — normalised seaborn heatmap
    4. per_class_accuracy  — bar chart sorted by accuracy
    5. confidence_histogram— softmax confidence: correct vs incorrect
    6. worst_samples       — most confidently wrong predictions

All figures are saved as high-quality PNGs to ``results/``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from utils.logger import get_logger

logger = get_logger(__name__)

# Consistent style across all plots
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      100,
})

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616])


def _denormalize(img_tensor: np.ndarray) -> np.ndarray:
    """Convert a normalised CHW image array back to HWC uint8."""
    img = img_tensor.transpose(1, 2, 0)          # CHW → HWC
    img = img * CIFAR10_STD + CIFAR10_MEAN       # denormalise
    img = np.clip(img, 0.0, 1.0)
    return img


class ResultPlotter:
    """Factory class that generates and saves all evaluation visualisations.

    Args:
        config:      Configuration dict from ``config/config.yaml``.
        class_names: List of 10 CIFAR-10 class name strings.
    """

    def __init__(self, config: dict, class_names: List[str]) -> None:
        self.config      = config
        self.class_names = class_names
        self.results_dir = config["visualization"]["results_dir"]
        self.dpi         = config["visualization"]["dpi"]
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Sample grid
    # ------------------------------------------------------------------

    def plot_sample_grid(self, data_loader: DataLoader) -> str:
        """Draw a 10-column × 5-row grid (one column per class).

        Args:
            data_loader: Any DataLoader over the CIFAR-10 dataset.

        Returns:
            Absolute path to the saved PNG file.
        """
        n_per_class = self.config["visualization"]["sample_grid"]["samples_per_class"]
        n_classes   = len(self.class_names)

        # Collect samples
        class_images: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_classes)}
        for images, labels in data_loader:
            for img, lbl in zip(images.numpy(), labels.numpy()):
                if len(class_images[lbl]) < n_per_class:
                    class_images[lbl].append(img)
            if all(len(v) >= n_per_class for v in class_images.values()):
                break

        fig, axes = plt.subplots(
            n_per_class, n_classes,
            figsize=(n_classes * 1.8, n_per_class * 1.8),
        )
        fig.suptitle("CIFAR-10 — Sample Images per Class", fontsize=14, y=1.01)

        for col, cls_name in enumerate(self.class_names):
            axes[0][col].set_title(cls_name, fontsize=9, fontweight="bold")
            for row in range(n_per_class):
                ax = axes[row][col]
                ax.imshow(_denormalize(class_images[col][row]))
                ax.axis("off")

        plt.tight_layout()
        path = self._save(fig, "01_sample_grid.png")
        return path

    # ------------------------------------------------------------------
    # 2. Training curves
    # ------------------------------------------------------------------

    def plot_training_curves(self, history: Dict[str, List[float]]) -> str:
        """Dual-subplot: loss curve and accuracy curve.

        Args:
            history: Dict with keys train_loss, val_loss, train_acc, val_acc.

        Returns:
            Absolute path to the saved PNG file.
        """
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Training & Validation Metrics", fontsize=14, fontweight="bold")

        # Loss
        ax1.plot(epochs, history["train_loss"], label="Train Loss",
                 color="#E74C3C", linewidth=2)
        ax1.plot(epochs, history["val_loss"],   label="Val Loss",
                 color="#3498DB", linewidth=2, linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curve")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(1, len(epochs))

        # Accuracy
        ax2.plot(epochs, history["train_acc"], label="Train Acc",
                 color="#E74C3C", linewidth=2)
        ax2.plot(epochs, history["val_acc"],   label="Val Acc",
                 color="#3498DB", linewidth=2, linestyle="--")
        best_epoch = int(np.argmax(history["val_acc"])) + 1
        best_acc   = max(history["val_acc"])
        ax2.axvline(best_epoch, color="grey", linestyle=":", alpha=0.7)
        ax2.annotate(
            f"Best: {best_acc:.1f}%\n(epoch {best_epoch})",
            xy=(best_epoch, best_acc),
            xytext=(best_epoch + 2, best_acc - 5),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="grey"),
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy Curve")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_xlim(1, len(epochs))
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        return self._save(fig, "02_training_curves.png")

    # ------------------------------------------------------------------
    # 3. Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, cm: np.ndarray) -> str:
        """Normalised seaborn heatmap of the 10×10 confusion matrix.

        Args:
            cm: Raw confusion matrix from sklearn (shape 10×10).

        Returns:
            Absolute path to the saved PNG file.
        """
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap="Blues", linewidths=0.4,
            linecolor="lightgrey", ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title("Normalised Confusion Matrix", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        return self._save(fig, "03_confusion_matrix.png")

    # ------------------------------------------------------------------
    # 4. Per-class accuracy bar chart
    # ------------------------------------------------------------------

    def plot_per_class_accuracy(self, cm: np.ndarray) -> str:
        """Horizontal bar chart of per-class accuracy, sorted descending.

        Each bar is colour-coded: green ≥ 90%, amber 80–90%, red < 80%.

        Args:
            cm: Raw confusion matrix (shape 10×10).

        Returns:
            Absolute path to the saved PNG file.
        """
        per_class_acc = 100.0 * cm.diagonal() / cm.sum(axis=1)
        order = np.argsort(per_class_acc)[::-1]
        names = [self.class_names[i] for i in order]
        accs  = per_class_acc[order]

        colors = ["#2ECC71" if a >= 90 else ("#F39C12" if a >= 80 else "#E74C3C")
                  for a in accs]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.6)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title("Per-Class Test Accuracy (sorted)", fontsize=13, fontweight="bold")
        ax.axvline(accs.mean(), color="grey", linestyle="--", alpha=0.7,
                   label=f"Mean: {accs.mean():.1f}%")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.1f}%", va="center", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        return self._save(fig, "04_per_class_accuracy.png")

    # ------------------------------------------------------------------
    # 5. Confidence histogram
    # ------------------------------------------------------------------

    def plot_confidence_histogram(
        self,
        all_probs:  np.ndarray,
        all_preds:  np.ndarray,
        all_labels: np.ndarray,
    ) -> str:
        """Histogram of max softmax confidence for correct vs incorrect predictions.

        Args:
            all_probs:  Softmax probability arrays (N, 10).
            all_preds:  Predicted label indices (N,).
            all_labels: Ground-truth label indices (N,).

        Returns:
            Absolute path to the saved PNG file.
        """
        max_conf = all_probs.max(axis=1)
        correct  = all_preds == all_labels

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(max_conf[correct],  bins=40, alpha=0.7, color="#2ECC71",
                label="Correct predictions",   density=True, edgecolor="white")
        ax.hist(max_conf[~correct], bins=40, alpha=0.7, color="#E74C3C",
                label="Incorrect predictions", density=True, edgecolor="white")
        ax.set_xlabel("Max Softmax Confidence")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return self._save(fig, "05_confidence_histogram.png")

    # ------------------------------------------------------------------
    # 6. Worst samples
    # ------------------------------------------------------------------

    def plot_worst_samples(self, worst_samples: List[Dict]) -> str:
        """Display top-K most confidently wrong predictions.

        Args:
            worst_samples: List of dicts from :meth:`Evaluator._worst_samples`.

        Returns:
            Absolute path to the saved PNG file.
        """
        k = len(worst_samples)
        if k == 0:
            logger.warning("No wrong samples to plot.")
            return ""

        fig, axes = plt.subplots(1, k, figsize=(k * 2.5, 3.5))
        if k == 1:
            axes = [axes]
        fig.suptitle("Most Confidently Wrong Predictions", fontsize=13, fontweight="bold")

        for ax, sample in zip(axes, worst_samples):
            img = _denormalize(sample["image"])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"True: {sample['true_label']}\n"
                f"Pred: {sample['pred_label']}\n"
                f"Conf: {sample['confidence']:.1%}",
                fontsize=8, color="#E74C3C",
            )
        plt.tight_layout()
        return self._save(fig, "06_worst_samples.png")

    # ------------------------------------------------------------------
    # 7. Model comparison
    # ------------------------------------------------------------------

    def plot_model_comparison(self, results_dict: Dict[str, Dict]) -> str:
        """Bar chart comparing test accuracy and F1 across multiple models.

        Args:
            results_dict: Mapping of model_name → evaluation results dict.
                          Each value must have keys ``accuracy`` and ``macro_f1``.

        Returns:
            Absolute path to the saved PNG file.
        """
        model_names = list(results_dict.keys())
        accuracies  = [results_dict[m]["accuracy"]  for m in model_names]
        macro_f1s   = [results_dict[m]["macro_f1"] * 100 for m in model_names]

        x     = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        bars1 = ax.bar(x - width / 2, accuracies, width, label="Test Accuracy (%)",
                       color="#3498DB", edgecolor="white")
        bars2 = ax.bar(x + width / 2, macro_f1s,  width, label="Macro F1 × 100",
                       color="#E74C3C", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel("Score (%)")
        ax.set_title("Model Comparison: Accuracy vs Macro F1", fontsize=13, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        return self._save(fig, "07_model_comparison.png")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> str:
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved plot → %s", path)
        return path
