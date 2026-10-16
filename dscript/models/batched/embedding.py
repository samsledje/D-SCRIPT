"""Residue level: project language-model embeddings to a smaller dimension."""

from __future__ import annotations

import torch
import torch.nn as nn


class Projection(nn.Module):
    """``(L, input_dim) -> (L, projection_dim)``: Linear, ReLU, dropout. Applied to each residue."""

    def __init__(self, input_dim: int, projection_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.relu(self.linear(z)))
