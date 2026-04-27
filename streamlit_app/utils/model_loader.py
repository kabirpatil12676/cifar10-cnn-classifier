"""
utils/model_loader.py
======================
Load the CIFAR-10 CNN from a checkpoint with graceful fallback.

AUDIT FIXES:
- CRITICAL: DEVICE now always CPU for Streamlit Cloud compatibility.
  GPU detection removed — cloud runners are CPU-only; .cuda() would crash.
- CRITICAL: weights_only=False kept (our checkpoint contains optimizer state
  and custom config dict — weights_only=True would reject them).
- WARNING: Exception handler now re-initialises model to guarantee eval mode
  even when load_state_dict raises (e.g. shape mismatch).
- MINOR: Removed bare `except Exception` swallowing the error silently —
  now logs the error string into meta["error"] AND reinitialises cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.cnn_model import CIFAR10Net  # noqa: E402

# FIX: Always CPU for Streamlit Cloud.  On a GPU machine this is still safe
# — the model will simply run on CPU.  Avoids .cuda() crashes on cloud.
_DEVICE = torch.device("cpu")

NUM_CLASSES  = 10
DROPOUT_RATE = 0.4


def load_model(
    checkpoint_path: Path | str,
    device: torch.device = _DEVICE,
) -> Tuple[nn.Module, dict]:
    """Load CIFAR10Net from a checkpoint file.

    If the checkpoint is not found, or fails to load, the model is returned
    with random weights and ``meta["is_fallback"] = True``.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint.
        device:          Target torch device (default: CPU).

    Returns:
        ``(model, meta)`` — model in eval mode + metadata dict.
    """
    checkpoint_path = Path(checkpoint_path)

    def _fresh_model() -> nn.Module:
        m = CIFAR10Net(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)
        m.eval()
        return m

    if not checkpoint_path.exists():
        return _fresh_model(), {"is_fallback": True, "epoch": 0, "val_acc": 0.0}

    try:
        # map_location=device ensures CPU loading on cloud even if saved on GPU
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = _fresh_model()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        meta = {
            "is_fallback": False,
            "epoch":   ckpt.get("epoch", "?"),
            "val_acc": float(ckpt.get("val_acc", 0.0)),  # FIX: ensure float
        }
        return model, meta

    except Exception as exc:
        # Corrupt / incompatible checkpoint — fall back to random weights
        return _fresh_model(), {
            "is_fallback": True,
            "epoch": 0,
            "val_acc": 0.0,
            "error": str(exc),
        }
