"""
Pair level: features for every residue pair, and the contact map built from them.

Both act on one cell at a time except the contact conv, which reaches
``kernel_width // 2`` cells each way - what lets the contact map be computed
in row tiles (see ``DScriptBatched.contact_map``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairFeatures(nn.Module):
    """
    ``h = LayerNorm(ReLU(W_sq (a-b)^2 + W_mul (a*b) + bias))`` over channels,
    computed from ``(a-b)^2 = a^2 + b^2 - 2ab`` without building the
    ``(b, 2d, N, M)`` input tensor.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_sq = nn.Parameter(torch.empty(hidden_dim, dim))
        self.w_mul = nn.Parameter(torch.empty(hidden_dim, dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        bound = 1.0 / (2 * dim) ** 0.5
        nn.init.uniform_(self.w_sq, -bound, bound)
        nn.init.uniform_(self.w_mul, -bound, bound)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``(b, N, d)``, ``(b, M, d)`` -> ``(b, h, N, M)``"""
        sq_a = (a * a) @ self.w_sq.t()  # (b, N, h)
        sq_b = (b * b) @ self.w_sq.t()  # (b, M, h)
        w_eff = self.w_mul - 2 * self.w_sq  # (h, d)
        bil = (a.unsqueeze(1) * w_eff[None, :, None, :]) @ b.transpose(1, 2).unsqueeze(1)
        h = bil + sq_a.transpose(1, 2).unsqueeze(-1) + sq_b.transpose(1, 2).unsqueeze(-2) + self.bias[None, :, None, None]
        h = F.relu(h)
        return self.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ContactHead(nn.Module):
    """Symmetric width-w conv, a learned affine, then a sigmoid."""

    def __init__(self, hidden_dim: int, width: int):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.clip()

    def clip(self) -> None:
        """Make the kernel equal its transpose, so the map transposes with the pair."""
        w = self.conv.weight.data
        w.copy_(0.5 * (w + w.transpose(2, 3)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(h) * self.scale + self.shift)
