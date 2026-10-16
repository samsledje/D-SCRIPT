"""Residue level: project language-model embeddings to a smaller dimension."""

from __future__ import annotations

import torch
import torch.nn as nn


class FullyConnectedEmbed(nn.Module):
    """
    Projection module: ``(b, N, input_dim) -> (b, N, projection_dim)`` with a
    linear layer, an activation, and dropout. Applied to each residue.
    """

    def __init__(self, nin: int, nout: int, dropout: float = 0.5, activation: nn.Module | None = None):
        super().__init__()
        self.transform = nn.Linear(nin, nout)
        self.activation = activation if activation is not None else nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.activation(self.transform(x)))
