"""
training/losses.py
==================
Custom loss functions for CIFAR-10 training.

Label Smoothing Cross-Entropy prevents the model from becoming
overconfident by distributing a small probability mass (epsilon)
uniformly across all non-target classes during training.

Reference: Szegedy et al., "Rethinking the Inception Architecture" (2016).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    For a target label ``y`` and smoothing factor ``smoothing``, the
    soft target distribution becomes::

        q(k) = (1 - smoothing) if k == y else smoothing / (num_classes - 1)

    This prevents the network from driving logits for the correct class to
    ``+inf``, which improves calibration and generalisation.

    Args:
        smoothing:   Label smoothing factor ``ε ∈ [0, 1)``.
                     Set to ``0.0`` to recover standard cross-entropy.
        reduction:   ``"mean"`` (default) or ``"sum"``.

    Raises:
        ValueError: If ``smoothing`` is not in ``[0, 1)``.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(
                f"smoothing must be in [0, 1), got {smoothing}"
            )
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the label-smoothed cross-entropy loss.

        Args:
            logits:  Raw model outputs ``(B, C)`` — no softmax applied.
            targets: Integer class indices ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        num_classes = logits.size(-1)
        log_probs   = F.log_softmax(logits, dim=-1)

        # Soft targets: uniform over all classes, then sharpen correct class
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs,
                                             self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                    1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
