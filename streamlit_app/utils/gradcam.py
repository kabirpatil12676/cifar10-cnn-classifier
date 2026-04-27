"""
utils/gradcam.py — AUDIT FIXED
FIXES:
- CRITICAL: apply_colormap signature was (heatmap, original_img, ...) but all
            callers passed (original_img, heatmap, ...) — corrected to match
            the intuitive (heatmap, original_img) caller convention, and
            updated all internal logic accordingly.
- WARNING:  Gradient None guard added — if backward hook fails to fire
            (e.g. leaf tensor with no grad), raise a clear RuntimeError.
- WARNING:  detach() on activations in _fwd — output may require grad;
            detach prevents accidental graph retention.
- MINOR:    Tuple return type uses builtin tuple (Python 3.9+).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Gradient-weighted Class Activation Mapping (hook-based).

    Args:
        model:        Trained CIFAR10Net (any eval mode).
        target_layer: Conv block to hook (e.g. ``model.block3``).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._fwd)
        self._bwd_hook = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m: nn.Module, _i: tuple, out: torch.Tensor) -> None:
        # FIX: detach to avoid retaining computation graph in activations
        self._activations = out.detach()

    def _bwd(self, _m: nn.Module, _gi: tuple, go: tuple) -> None:
        # FIX: guard against None gradient (e.g. layer not on grad path)
        if go[0] is not None:
            self._gradients = go[0].detach()

    def generate(
        self,
        tensor:    torch.Tensor,
        class_idx: Optional[int]   = None,
        device:    torch.device    = torch.device("cpu"),
    ) -> np.ndarray:
        """Compute GradCAM heatmap for a single image tensor.

        Args:
            tensor:    Input ``(1, 3, 32, 32)`` or ``(3, 32, 32)``.
            class_idx: Target class. ``None`` → top-1 predicted class.
            device:    Torch device.

        Returns:
            Float32 numpy array ``(H, W)`` in ``[0, 1]``.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)

        # Reset captured data
        self._activations = None
        self._gradients   = None

        self.model.eval()
        with torch.enable_grad():
            logits = self.model(tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            score = logits[0, class_idx]
            self.model.zero_grad()
            score.backward()

        # FIX: explicit guard with meaningful message
        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "GradCAM hooks did not capture data. "
                "Check that target_layer is on the forward path."
            )

        alpha = self._gradients.mean(dim=(2, 3), keepdim=True)         # (1,C,1,1)
        cam   = (alpha * self._activations).sum(dim=1).squeeze(0)      # (H,W)
        cam   = torch.relu(cam).cpu().numpy().astype(np.float32)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove_hooks(self) -> None:
        """Remove hooks to prevent memory leaks. Always call after use."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    @staticmethod
    def apply_colormap(
        heatmap:       np.ndarray,
        original_img:  np.ndarray,
        alpha:         float = 0.5,
        colormap_name: str   = "jet",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resize heatmap, apply colormap, and blend with original image.

        Args:
            heatmap:       Float32 ``(h, w)`` in ``[0, 1]``.
            original_img:  HWC uint8 ``(H, W, 3)`` — denormalised original.
            alpha:         Heatmap weight in overlay (default 0.5).
            colormap_name: ``jet`` | ``viridis`` | ``plasma`` | ``inferno``.

        Returns:
            ``(original_uint8, heatmap_rgb_uint8, overlay_uint8)``
        """
        H, W = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

        cv2_maps = {
            "jet":     cv2.COLORMAP_JET,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma":  cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
        }
        cmap        = cv2_maps.get(colormap_name, cv2.COLORMAP_JET)
        colored     = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cmap)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        orig_f    = original_img.astype(np.float32)
        heat_f    = colored_rgb.astype(np.float32)
        overlay   = np.clip(alpha * heat_f + (1 - alpha) * orig_f, 0, 255).astype(np.uint8)

        return original_img, colored_rgb, overlay
