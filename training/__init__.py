"""Training package for CIFAR-10 classifier."""
from training.trainer import Trainer
from training.losses import LabelSmoothingCrossEntropy

__all__ = ["Trainer", "LabelSmoothingCrossEntropy"]
