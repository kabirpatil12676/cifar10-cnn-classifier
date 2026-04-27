"""
models/cnn_model.py
===================
Custom CNN with ResNet-inspired residual skip connections for CIFAR-10.

Architecture overview:
    Block 1 : ResidualBlock(3  → 64)  + MaxPool2d(2)   → (B, 64,  16, 16)
    Block 2 : ResidualBlock(64 → 128) + MaxPool2d(2)   → (B, 128,  8,  8)
    Block 3 : ResidualBlock(128→ 256) + AdaptiveAvgPool → (B, 256,  2,  2)
    Head    : Flatten → Linear(1024→512) → ReLU → Dropout → Linear(512→10)

Total trainable parameters: ~2.8 M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two-layer residual block with optional projection shortcut.

    If ``in_channels != out_channels``, a 1×1 convolution projects the
    identity to the correct depth so the skip-addition is dimension-safe
    (identical to the ResNet "option B" shortcut).

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection shortcut (only when channel counts differ)
        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(B, C_in, H, W)``.

        Returns:
            Output tensor ``(B, C_out, H, W)`` — same spatial size as input.
        """
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out = out + identity          # residual addition
        out = F.relu(out, inplace=True)
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CIFAR10Net(nn.Module):
    """Custom ResNet-inspired CNN for CIFAR-10 (32×32 RGB input).

    Three residual blocks progressively double the channel depth while
    halving spatial resolution, followed by a small fully-connected head.
    Kaiming He weight initialisation is applied to all Conv2d / Linear layers.

    Args:
        num_classes:  Number of output logits (default: 10).
        dropout_rate: Dropout probability before the final linear layer
                      (default: 0.4).
    """

    def __init__(
        self,
        num_classes:  int   = 10,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()

        # --- Feature extractor -------------------------------------------
        # Block 1: (B, 3, 32, 32) → (B, 64, 16, 16)
        self.block1 = ResidualBlock(3, 64)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: (B, 64, 16, 16) → (B, 128, 8, 8)
        self.block2 = ResidualBlock(64, 128)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: (B, 128, 8, 8) → (B, 256, 2, 2)
        self.block3 = ResidualBlock(128, 256)
        self.pool3  = nn.AdaptiveAvgPool2d((2, 2))

        # --- Classification head -----------------------------------------
        self.flatten  = nn.Flatten()                    # → (B, 1024)
        self.fc1      = nn.Linear(256 * 2 * 2, 512)
        self.dropout  = nn.Dropout(p=dropout_rate)
        self.fc2      = nn.Linear(512, num_classes)

        self._initialize_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and head.

        Args:
            x: Input image batch ``(B, 3, 32, 32)``.

        Returns:
            Raw logits ``(B, num_classes)`` — no softmax applied.
        """
        # Feature extraction
        x = self.pool1(self.block1(x))   # (B,  64, 16, 16)
        x = self.pool2(self.block2(x))   # (B, 128,  8,  8)
        x = self.pool3(self.block3(x))   # (B, 256,  2,  2)

        # Classification head
        x = self.flatten(x)              # (B, 1024)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)                  # (B, 10)
        return x

    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        """Apply Kaiming He initialisation to Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_model(config: dict) -> CIFAR10Net:
    """Instantiate ``CIFAR10Net`` from a configuration dictionary.

    Args:
        config: Dict loaded from ``config/config.yaml``.

    Returns:
        An initialised :class:`CIFAR10Net` instance.
    """
    model_cfg = config["model"]
    return CIFAR10Net(
        num_classes=model_cfg["num_classes"],
        dropout_rate=model_cfg["dropout_rate"],
    )
