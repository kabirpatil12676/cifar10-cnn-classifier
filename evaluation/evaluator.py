"""
evaluation/evaluator.py
========================
Comprehensive model evaluation for CIFAR-10.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Runs full evaluation of a trained CIFAR-10 model.

    Args:
        model:        Trained CIFAR10Net instance.
        data_loader:  DataLoader for the test or val split.
        class_names:  List of 10 CIFAR-10 class name strings.
        device:       Torch device.
    """

    def __init__(self, model: nn.Module, data_loader: DataLoader,
                 class_names: List[str], device: torch.device) -> None:
        self.model = model.to(device)
        self.data_loader = data_loader
        self.class_names = class_names
        self.device = device

    def evaluate(self) -> Dict:
        """Run full evaluation and return metrics dictionary.

        Returns:
            Dict with keys: accuracy, report, per_class, confusion_matrix,
            macro_f1, weighted_f1, worst_samples, all_probs, all_preds, all_labels.
        """
        logger.info("Running evaluation on %d batches …", len(self.data_loader))
        all_preds, all_labels, all_probs, all_images = self._collect_predictions()

        accuracy = float(100.0 * (all_preds == all_labels).mean())
        report = classification_report(
            all_labels, all_preds, target_names=self.class_names, digits=4
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=list(range(10))
        )
        per_class = {
            self.class_names[i]: {
                "precision": float(precision[i]),
                "recall":    float(recall[i]),
                "f1":        float(f1[i]),
            }
            for i in range(10)
        }
        _, _, macro_f1, _    = precision_recall_fscore_support(all_labels, all_preds, average="macro")
        _, _, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
        cm    = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
        worst = self._worst_samples(all_images, all_labels, all_preds, all_probs)

        results = {
            "accuracy": accuracy, "report": report, "per_class": per_class,
            "confusion_matrix": cm, "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1), "worst_samples": worst,
            "all_probs": all_probs, "all_preds": all_preds, "all_labels": all_labels,
        }
        logger.info("Overall accuracy: %.2f%% | Macro F1: %.4f", accuracy, macro_f1)
        logger.info("\n%s", report)
        return results

    def _collect_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Inference over the full DataLoader."""
        self.model.eval()
        preds_list, labels_list, probs_list, images_list = [], [], [], []

        with torch.no_grad():
            for images, labels in self.data_loader:
                images_gpu = images.to(self.device, non_blocking=True)
                logits     = self.model(images_gpu)
                probs      = torch.softmax(logits, dim=1)
                preds      = probs.argmax(dim=1)
                preds_list.append(preds.cpu().numpy())
                labels_list.append(labels.numpy())
                probs_list.append(probs.cpu().numpy())
                images_list.append(images.numpy())

        return (
            np.concatenate(preds_list), np.concatenate(labels_list),
            np.concatenate(probs_list), np.concatenate(images_list),
        )

    def _worst_samples(self, images: np.ndarray, labels: np.ndarray,
                       preds: np.ndarray, probs: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Return top-K samples where the model was most confidently wrong."""
        wrong_mask        = preds != labels
        wrong_indices     = np.where(wrong_mask)[0]
        if len(wrong_indices) == 0:
            return []
        wrong_confidences = probs[wrong_indices, preds[wrong_indices]]
        sorted_order      = np.argsort(wrong_confidences)[::-1]
        top_wrong         = wrong_indices[sorted_order[:top_k]]
        return [
            {"image": images[i], "true_label": self.class_names[labels[i]],
             "pred_label": self.class_names[preds[i]], "confidence": float(probs[i, preds[i]])}
            for i in top_wrong
        ]


def load_model_for_eval(checkpoint_path: str, model: nn.Module,
                         device: torch.device) -> nn.Module:
    """Load checkpoint into model and return in eval mode."""
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint | epoch=%s | val_acc=%.2f%%",
                checkpoint.get("epoch", "?"), checkpoint.get("val_acc", float("nan")))
    return model
