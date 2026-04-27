"""
utils/seed.py
=============
Reproducibility helpers — seed every source of randomness.

Seeding torch, numpy, random, and CUDA ensures that experiments are
fully reproducible when run with the same seed, hardware, and
``torch.backends.cudnn`` settings.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all sources of randomness for full reproducibility.

    Sets seeds for:
        - Python built-in :mod:`random`
        - :mod:`numpy`
        - :mod:`torch` (CPU and CUDA)
        - CUDA cuDNN (deterministic mode)

    Args:
        seed: Integer seed value (default: 42).

    Note:
        ``torch.backends.cudnn.benchmark = False`` may slow down training
        slightly but is required for strict reproducibility on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)          # multi-GPU

    # Deterministic cuDNN ops (slight perf cost, ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
