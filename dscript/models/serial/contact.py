"""
Pair level: features for every residue pair, and the contact map built from them.

This is the original D-SCRIPT contact module. Its BatchNorms normalise over
the whole contact map of the pair being processed, so in training a pair's
result depends on its entire map: the map cannot be split into tiles, and the
model runs one pair at a time (see ``DScriptSerial.forward``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    """
    Pair features: ``[|z0 - z1|, z0 * z1]`` for every residue pair, a 1x1 conv
    to ``hidden_dim`` channels, an activation, and BatchNorm.
    ``(b, N, d), (b, M, d) -> (b, h, N, M)``.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, activation: nn.Module | None = None):
        super().__init__()
        self.conv = nn.Conv2d(2 * embed_dim, hidden_dim, 1)
        self.batchnorm = nn.BatchNorm2d(hidden_dim)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        z0 = z0.transpose(1, 2)  # (b, d, N)
        z1 = z1.transpose(1, 2)  # (b, d, M)
        z_dif = torch.abs(z0.unsqueeze(3) - z1.unsqueeze(2))
        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        c = self.conv(torch.cat([z_dif, z_mul], 1))
        return self.batchnorm(self.activation(c))


class ContactCNN(nn.Module):
    """
    Contact map: pair features (``hidden``), then a symmetric width-w conv,
    BatchNorm, and a sigmoid. ``(b, N, d), (b, M, d) -> (b, 1, N, M)``.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 50, width: int = 7, activation: nn.Module | None = None):
        super().__init__()
        self.hidden = FullyConnected(embed_dim, hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation if activation is not None else nn.Sigmoid()
        self.clip()

    def clip(self) -> None:
        """Make the kernel equal its transpose, so the map transposes with the pair."""
        w = self.conv.weight.data
        w.copy_(0.5 * (w + w.transpose(2, 3)))

    def cmap(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Pair features, ``(b, h, N, M)``."""
        return self.hidden(z0, z1)

    def predict(self, C: torch.Tensor) -> torch.Tensor:
        """Contact map from pair features, ``(b, 1, N, M)``."""
        return self.activation(self.batchnorm(self.conv(C)))

    def forward(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return self.predict(self.cmap(z0, z1))
