"""
visualization/gradcam.py
=========================
GradCAM (Gradient-weighted Class Activation Mapping) implementation.

GradCAM computes a coarse localisation map that highlights the regions
of an input image that are most important for the model's prediction.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Usage:
    gradcam = GradCAM(model, target_layer=model.block3)
    heatmap = gradcam.generate(image_tensor, class_idx=None)  # None = predicted class
    gradcam.plot_overlay(original_img, heatmap, save_path="results/gradcam.png")
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import get_logger

logger = get_logger(__name__)

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616])


class GradCAM:
    """Gradient-weighted Class Activation Mapping for PyTorch CNN models.

    Registers forward and backward hooks on ``target_layer`` to capture:
        - Feature maps (activations) during the forward pass.
        - Gradients w.r.t. those feature maps during backprop.

    The CAM is computed as:
        cam = ReLU( sum_k( alpha_k * A_k ) )
    where alpha_k = global_average_pool( dY/dA_k ).

    Args:
        model:        The trained CNN (must be in eval mode for inference).
        target_layer: The ``nn.Module`` layer to hook (typically the last
                      conv block before global pooling).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _save_activations(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> None:
        self._activations = output.detach()

    def _save_gradients(
        self, module: nn.Module, grad_input: tuple, grad_output: tuple
    ) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        """Generate the GradCAM heatmap for a single image.

        Args:
            image_tensor: Pre-processed image tensor of shape ``(1, C, H, W)``
                          or ``(C, H, W)`` (batch dim added automatically).
            class_idx:    Target class index. If ``None``, uses the model's
                          top-1 predicted class.
            device:       Torch device.

        Returns:
            Normalised heatmap as a float32 numpy array of shape ``(H, W)``
            with values in ``[0, 1]``.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device).requires_grad_(False)

        self.model.eval()

        # Forward pass — keep graph for backward
        with torch.enable_grad():
            logits = self.model(image_tensor)              # (1, 10)

            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            # Scalar score for the target class
            score = logits[0, class_idx]

            self.model.zero_grad()
            score.backward()

        # alpha_k = GAP over spatial dims of gradients: shape (C,)
        gradients   = self._gradients   # (1, C, h, w)
        activations = self._activations # (1, C, h, w)

        if gradients is None or activations is None:
            raise RuntimeError("Hooks did not capture data — check target_layer.")

        alpha  = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam    = (alpha * activations).sum(dim=1).squeeze(0)  # (h, w)
        cam    = torch.relu(cam).cpu().numpy()

        # Normalise
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove registered hooks (call when done to avoid memory leaks)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def overlay_heatmap(
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay a GradCAM heatmap on the original image.

        Args:
            original_img: HWC float image in ``[0, 1]`` (denormalised).
            heatmap:      Float32 array of shape ``(h, w)`` in ``[0, 1]``.
            alpha:        Heatmap opacity (0 = image only, 1 = heatmap only).

        Returns:
            Blended HWC uint8 image.
        """
        h, w = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB) / 255.0
        overlay  = alpha * colormap + (1 - alpha) * original_img
        overlay  = np.clip(overlay, 0, 1)
        return (overlay * 255).astype(np.uint8)

    def generate_batch_plots(
        self,
        data_loader: DataLoader,
        class_names: List[str],
        config: dict,
        device: torch.device,
        n_images: int = 10,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate a grid of GradCAM overlays — one per CIFAR-10 class.

        Selects the first correctly-predicted example for each class from
        ``data_loader`` and displays the original image alongside its CAM.

        Args:
            data_loader: DataLoader for the test split.
            class_names: List of class names.
            config:      Config dict (used for ``alpha`` and ``results_dir``).
            device:      Torch device.
            n_images:    Number of images to plot (default: 10).
            save_path:   Override save location; defaults to results_dir.

        Returns:
            Path to the saved PNG.
        """
        alpha       = config["visualization"]["gradcam"]["alpha"]
        results_dir = config["visualization"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)

        # Collect one correct sample per class
        collected: dict[int, tuple] = {}  # cls_idx → (image_tensor, cam)
        self.model.eval()

        for images, labels in data_loader:
            if len(collected) >= n_images:
                break
            images_gpu = images.to(device)
            with torch.no_grad():
                preds = self.model(images_gpu).argmax(dim=1).cpu()

            for i, (img, lbl, pred) in enumerate(zip(images, labels, preds)):
                cls = lbl.item()
                if cls not in collected and pred.item() == cls:
                    cam = self.generate(img, class_idx=cls, device=device)
                    collected[cls] = (img, cam)

        classes_found = sorted(collected.keys())
        n = len(classes_found)
        if n == 0:
            logger.warning("No samples collected for GradCAM.")
            return ""

        fig, axes = plt.subplots(2, n, figsize=(n * 2.2, 5))
        fig.suptitle("GradCAM — What Does the Model Focus On?",
                     fontsize=13, fontweight="bold")

        for col, cls_idx in enumerate(classes_found):
            img_t, cam = collected[cls_idx]
            # Denormalise original
            orig = img_t.numpy().transpose(1, 2, 0)
            orig = orig * CIFAR10_STD + CIFAR10_MEAN
            orig = np.clip(orig, 0, 1)

            overlay = self.overlay_heatmap(orig, cam, alpha=alpha)

            axes[0][col].imshow(orig)
            axes[0][col].axis("off")
            axes[0][col].set_title(class_names[cls_idx], fontsize=8, fontweight="bold")

            axes[1][col].imshow(overlay)
            axes[1][col].axis("off")

        axes[0][0].set_ylabel("Original", fontsize=9, rotation=90, labelpad=4)
        axes[1][0].set_ylabel("GradCAM",  fontsize=9, rotation=90, labelpad=4)

        plt.tight_layout()
        out_path = save_path or os.path.join(results_dir, "08_gradcam_heatmaps.png")
        fig.savefig(out_path, dpi=config["visualization"]["dpi"], bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved GradCAM grid → %s", out_path)
        return out_path
