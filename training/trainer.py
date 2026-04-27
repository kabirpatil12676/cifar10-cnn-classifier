"""
training/trainer.py
===================
Full training loop with:
    - Mixed-precision training (torch.cuda.amp)
    - CosineAnnealingLR scheduler
    - Early stopping (patience-based)
    - Per-epoch metric logging (loss, accuracy, lr)
    - Best-model checkpointing
    - Graceful keyboard-interrupt handling
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from training.losses import LabelSmoothingCrossEntropy
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------

class TrainingHistory:
    """Accumulates per-epoch metrics for later plotting.

    Attributes:
        train_loss: List of average training losses per epoch.
        val_loss:   List of average validation losses per epoch.
        train_acc:  List of training accuracies (0–100) per epoch.
        val_acc:    List of validation accuracies (0–100) per epoch.
        lr:         Learning rate at the end of each epoch.
    """

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.val_loss:   List[float] = []
        self.train_acc:  List[float] = []
        self.val_acc:    List[float] = []
        self.lr:         List[float] = []

    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Append one epoch's metrics."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.lr.append(lr)

    def as_dict(self) -> Dict[str, List[float]]:
        """Return all metrics as a plain dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss":   self.val_loss,
            "train_acc":  self.train_acc,
            "val_acc":    self.val_acc,
            "lr":         self.lr,
        }


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation accuracy stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        delta:    Minimum improvement required to reset the counter.
    """

    def __init__(self, patience: int = 15, delta: float = 1e-4) -> None:
        self.patience      = patience
        self.delta         = delta
        self._counter      = 0
        self._best_score:  Optional[float] = None
        self.should_stop   = False

    def step(self, val_acc: float) -> bool:
        """Update state based on latest validation accuracy.

        Args:
            val_acc: Validation accuracy for the current epoch (0–100).

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        score = val_acc
        if self._best_score is None:
            self._best_score = score
        elif score < self._best_score + self.delta:
            self._counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d / %d epochs",
                self._counter, self.patience,
            )
            if self._counter >= self.patience:
                self.should_stop = True
        else:
            self._best_score = score
            self._counter    = 0

        return self.should_stop


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Manages the complete training and validation cycle.

    Args:
        model:        The neural network to train.
        config:       Configuration dict from ``config/config.yaml``.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        device:       Torch device (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device: torch.device,
    ) -> None:
        self.model        = model.to(device)
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        tcfg = config["training"]

        # Loss
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=tcfg["label_smoothing"]
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=tcfg["learning_rate"],
            weight_decay=tcfg["weight_decay"],
        )

        # Scheduler
        scfg = tcfg["scheduler"]
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=scfg["T_max"],
            eta_min=scfg["eta_min"],
        )

        # Mixed precision scaler (no-op on CPU)
        # Uses the modern torch.amp API (torch.cuda.amp is deprecated in 2.x)
        self.use_amp    = tcfg["mixed_precision"] and torch.cuda.is_available()
        self._amp_dtype = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler     = torch.amp.GradScaler(self._amp_dtype, enabled=self.use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=tcfg["early_stopping_patience"]
        )

        # Checkpoint paths
        self.ckpt_dir      = tcfg["checkpoint_dir"]
        self.best_name     = tcfg["best_model_name"]
        self.last_name     = tcfg["last_model_name"]
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.epochs        = tcfg["epochs"]
        self.log_interval  = tcfg["log_interval"]
        self.best_val_acc  = 0.0
        self.history       = TrainingHistory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainingHistory:
        """Run the full training loop.

        Returns:
            A :class:`TrainingHistory` containing per-epoch metrics.
        """
        logger.info("Starting training | device=%s | epochs=%d | amp=%s",
                    self.device, self.epochs, self.use_amp)
        logger.info("Train batches: %d | Val batches: %d",
                    len(self.train_loader), len(self.val_loader))

        try:
            for epoch in range(1, self.epochs + 1):
                t0 = time.perf_counter()

                train_loss, train_acc = self._train_one_epoch(epoch)
                val_loss, val_acc     = self._validate()

                # Capture LR *before* stepping (this epoch's effective LR)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()

                elapsed = time.perf_counter() - t0

                logger.info(
                    "Epoch %3d/%d | "
                    "Train Loss: %.4f  Acc: %.2f%% | "
                    "Val Loss: %.4f  Acc: %.2f%% | "
                    "LR: %.2e | Time: %.1fs",
                    epoch, self.epochs,
                    train_loss, train_acc,
                    val_loss,   val_acc,
                    current_lr, elapsed,
                )

                self.history.update(train_loss, val_loss,
                                    train_acc, val_acc, current_lr)

                # Save best checkpoint
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self._save_checkpoint(self.best_name, epoch, val_acc)
                    logger.info("  ✓ New best model saved (val_acc=%.2f%%)", val_acc)

                # Early stopping check
                if self.early_stopping.step(val_acc):
                    logger.info(
                        "Early stopping triggered after epoch %d. "
                        "Best val_acc=%.2f%%", epoch, self.best_val_acc
                    )
                    break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")

        # Always persist last state
        self._save_checkpoint(self.last_name, epoch=-1, val_acc=self.best_val_acc)
        logger.info("Training complete. Best val_acc=%.2f%%", self.best_val_acc)
        return self.history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """Run one full pass over the training DataLoader.

        Args:
            epoch: Current epoch number (1-indexed) for log display.

        Returns:
            ``(avg_loss, accuracy_percent)`` over the training set.
        """
        self.model.train()
        running_loss    = 0.0
        correct         = 0
        total           = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader, 1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self._amp_dtype, enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

            if batch_idx % self.log_interval == 0:
                logger.debug(
                    "  Epoch %d | Batch %d/%d | Loss: %.4f",
                    epoch, batch_idx, len(self.train_loader),
                    running_loss / batch_idx,
                )

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def _validate(self) -> tuple[float, float]:
        """Evaluate the model on the validation DataLoader.

        Returns:
            ``(avg_loss, accuracy_percent)`` over the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct      = 0
        total        = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self._amp_dtype, enabled=self.use_amp):
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

                running_loss += loss.item()
                preds         = logits.argmax(dim=1)
                correct      += (preds == labels).sum().item()
                total        += labels.size(0)

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_acc: float,
    ) -> None:
        """Persist model state + metadata to disk.

        Args:
            filename: File name (not full path) for the checkpoint.
            epoch:    Current epoch number. Use -1 for final save.
            val_acc:  Validation accuracy at save time.
        """
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(
            {
                "epoch":      epoch,
                "val_acc":    val_acc,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config":     self.config,
            },
            path,
        )
