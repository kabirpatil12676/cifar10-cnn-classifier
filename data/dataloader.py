"""
data/dataloader.py
==================
Data loading, augmentation, and preprocessing pipeline for CIFAR-10.

Responsibilities:
    - Download CIFAR-10 via torchvision (auto-cached).
    - Build training augmentation transforms (crop, flip, color jitter).
    - Build evaluation transforms (normalize only).
    - Split the 50 000-sample training set into 45 000 train / 5 000 val
      using a reproducible seed — the 10 000-sample test set is untouched.
    - Return typed (DataLoader, DataLoader, DataLoader) tuple.

Usage:
    from data.dataloader import get_dataloaders, get_class_names
    train_loader, val_loader, test_loader = get_dataloaders(config)
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(config: dict, mode: str = "train") -> transforms.Compose:
    """Build the data-transformation pipeline for a given split.

    Training applies full augmentation; validation and test apply only
    tensor conversion and channel-wise normalization.

    Args:
        config: Configuration dict loaded from config.yaml.
        mode:   One of ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.

    Raises:
        ValueError: If ``mode`` is not one of the accepted values.
    """
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"mode must be 'train', 'val', or 'test'; got '{mode}'")

    mean: List[float] = config["data"]["mean"]
    std:  List[float] = config["data"]["std"]
    normalize = transforms.Normalize(mean=mean, std=std)

    if mode == "train":
        aug = config["data"]["augmentation"]
        jitter = aug["color_jitter"]
        pipeline = [
            transforms.RandomCrop(
                aug["random_crop_size"],
                padding=aug["random_crop_padding"],
            ),
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
            transforms.ColorJitter(
                brightness=jitter["brightness"],
                contrast=jitter["contrast"],
                saturation=jitter["saturation"],
                hue=jitter["hue"],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        pipeline = [
            transforms.ToTensor(),
            normalize,
        ]

    return transforms.Compose(pipeline)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    config: dict,
    data_root: str | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for CIFAR-10.

    The 50 000-sample training set is split (using a fixed NumPy seed from
    ``config["seed"]``) into:
        - **train** : 45 000 samples with augmentation transforms.
        - **val**   :  5 000 samples with eval-only transforms.
    The 10 000-sample test set is never used during training.

    Args:
        config:    Configuration dict loaded from ``config/config.yaml``.
        data_root: Optional override for the dataset root directory.
                   Defaults to ``config["data"]["root"]``.

    Returns:
        ``(train_loader, val_loader, test_loader)`` — a tuple of three
        :class:`torch.utils.data.DataLoader` instances.
    """
    root        = data_root or config["data"]["root"]
    batch_size  = config["data"]["batch_size"]
    pin_memory  = config["data"]["pin_memory"]
    val_split   = config["data"]["val_split"]
    seed        = config["seed"]

    # num_workers > 0 causes multiprocessing errors on Windows with Jupyter.
    # Auto-detect: use 0 on Windows, configured value elsewhere.
    cfg_workers  = config["data"]["num_workers"]
    num_workers  = 0 if sys.platform == "win32" else cfg_workers

    os.makedirs(root, exist_ok=True)

    # Download once — torchvision caches the files.
    full_train_aug  = CIFAR10(root=root, train=True,  download=True,
                               transform=get_transforms(config, "train"))
    full_train_eval = CIFAR10(root=root, train=True,  download=False,
                               transform=get_transforms(config, "val"))
    testset         = CIFAR10(root=root, train=False, download=True,
                               transform=get_transforms(config, "test"))

    # Reproducible index shuffle
    n_total  = len(full_train_aug)          # 50 000
    n_val    = int(n_total * val_split)     # 5 000
    n_train  = n_total - n_val             # 45 000

    rng     = np.random.default_rng(seed=seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    train_idx = indices[:n_train].tolist()
    val_idx   = indices[n_train:].tolist()

    trainset = Subset(full_train_aug,  train_idx)   # augmented
    valset   = Subset(full_train_eval, val_idx)     # clean (no augmentation)

    # Build loaders
    _kw_shared = dict(num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,       # keep batch sizes uniform for stable BN
        **_kw_shared,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        **_kw_shared,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        **_kw_shared,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_class_names(config: dict) -> List[str]:
    """Return the list of CIFAR-10 class label strings from config.

    Args:
        config: Configuration dict.

    Returns:
        A list of 10 class name strings in label-index order.
    """
    return list(config["classes"])
