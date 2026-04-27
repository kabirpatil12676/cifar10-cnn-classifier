"""
models/cnn_model.py
====================
CIFAR-10 CNN with ResNet-inspired residual skip connections.
Identical to the training repository — kept in sync intentionally.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-layer residual block with optional 1×1 projection shortcut.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CIFAR10Net(nn.Module):
    """Custom ResNet-inspired CNN for 32×32 CIFAR-10 images.

    Args:
        num_classes:  Number of output classes (default: 10).
        dropout_rate: Dropout probability in the classifier head (default: 0.4).
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.4) -> None:
        super().__init__()
        self.block1 = ResidualBlock(3, 64)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.block2 = ResidualBlock(64, 128)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.block3 = ResidualBlock(128, 256)
        self.pool3  = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(256 * 2 * 2, 512)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2     = nn.Linear(512, num_classes)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


def build_model(config: dict) -> CIFAR10Net:
    """Build model from config dict."""
    return CIFAR10Net(
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"],
    )
