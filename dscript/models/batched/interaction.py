"""
The batched D-SCRIPT model, trained by ``dscript train_batched``.

Projection (embedding.py) -> pair features and contact map (contact.py) ->
weighted, thresholded pooling (here). Every operation is per pair (no
BatchNorm), so padding is exact and results do not depend on batch size or
device count. The prediction is the same for (A, B) and (B, A).

The pair features have ``hidden_dim`` channels for every residue pair, the
dominant memory cost. The contact map is therefore computed in row tiles with
``kernel_width // 2`` overlapping rows, and in training each tile is
recomputed during backward instead of stored. Tiling never changes the result.

Saved models are single files; see ``models/saving.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...core.batch import PairBatch
from ..saving import load_into, save_model
from .contact import ContactHead, PairFeatures
from .embedding import Projection


def pad(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(L_i, d)`` tensors -> ``(b, L_max, d)`` and a ``(b, L_max)`` mask."""
    lengths = torch.tensor([s.shape[0] for s in seqs], device=seqs[0].device)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
    return padded, mask


class DScriptBatched(nn.Module):
    """
    ``forward(batch, tile_rows=None) -> {"p_hat": (b,), "cmap_magnitude": (b,)}``.

    :param input_dim: language-model embedding dimension
    :param features: per-residue features appended after projection,
        ``{name: width}`` in the order the batch concatenates them

    ``tile_rows`` bounds how many contact-map rows are computed at once; None
    computes the whole map in one tile. It changes memory and speed, never
    the result.
    """

    family = "batched"

    def __init__(
        self,
        input_dim: int,
        features: dict[str, int] | None = None,
        projection_dim: int = 100,
        hidden_dim: int = 50,
        kernel_width: int = 7,
        dropout: float = 0.5,
        weighting: bool = True,
        sigmoid: bool = True,
        pool_width: int | None = None,
        theta_init: float = 1.0,
        lambda_init: float = 0.0,
        gamma_init: float = 0.0,
        sigmoid_k: float = 20.0,
        sigmoid_x0: float = 0.5,
    ):
        super().__init__()
        if kernel_width % 2 == 0:
            raise ValueError("kernel_width must be odd")
        features = dict(features or {})
        # Constructor arguments, saved with the weights to rebuild the model.
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.features = features  # {name: width}, in concatenation order
        feature_dim = sum(features.values())
        self.projection = Projection(input_dim, projection_dim, dropout)
        self.pair = PairFeatures(projection_dim + feature_dim, hidden_dim)
        self.head = ContactHead(hidden_dim, kernel_width)
        self.halo = kernel_width // 2

        self.weighting = weighting
        self.sigmoid = sigmoid
        self.pool_width = pool_width
        self.sigmoid_k = sigmoid_k
        self.sigmoid_x0 = sigmoid_x0
        if weighting:
            self.theta = nn.Parameter(torch.tensor([float(theta_init)]))
            self.lambda_ = nn.Parameter(torch.tensor([float(lambda_init)]))
        self.gamma = nn.Parameter(torch.tensor([float(gamma_init)]))
        self.clip()

    def clip(self) -> None:
        """Project parameters back into their valid range; call after every optimizer step."""
        self.head.clip()
        if self.weighting:
            self.theta.data.clamp_(0, 1)
            self.lambda_.data.clamp_(min=0)
        self.gamma.data.clamp_(min=0)

    # -- residue level -------------------------------------------------------

    def _embed(self, zs: list[torch.Tensor], fs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Project every protein of one side in one packed call, append features, pad."""
        lengths = [z.shape[0] for z in zs]
        proj = self.projection(torch.cat(zs, dim=0)).split(lengths)
        per_protein = [torch.cat([p, f], dim=-1) for p, f in zip(proj, fs, strict=True)]
        return pad(per_protein)

    # -- contact map ---------------------------------------------------------

    def _tile(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.pair(a, b) * mask)

    def contact_map(self, e0, e1, pair_mask, tile_rows: int | None = None) -> torch.Tensor:
        """
        ``(b, 1, N, M)`` contact map, computed ``tile_rows`` rows at a time.

        Each tile is computed with ``halo`` extra rows on both sides, and
        those rows are dropped: the conv's zero padding at a tile's inner edge
        only affects halo rows. At the map's real edges there are no extra
        rows, which matches zero padding exactly.
        """
        n = e0.shape[1]
        rows = n if tile_rows is None else max(1, min(int(tile_rows), n))
        # Recomputing a tile in backward only saves memory when there is more
        # than one tile: with a single tile the whole map is live anyway, so
        # checkpointing there would buy nothing and cost an extra forward.
        recompute = rows < n and self.training and torch.is_grad_enabled()
        mask = pair_mask.to(e0.dtype)
        tiles = []
        for i in range(0, n, rows):
            j = min(i + rows, n)
            lo, hi = max(i - self.halo, 0), min(j + self.halo, n)
            args = (e0[:, lo:hi], e1, mask[:, :, lo:hi])
            c = checkpoint(self._tile, *args, use_reentrant=False) if recompute else self._tile(*args)
            tiles.append(c[:, :, i - lo : j - lo])
        return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=2)

    # -- pooling -------------------------------------------------------------

    def _weight_matrix(self, n_valid, m_valid, n_pad, m_pad, like):
        idx_n = torch.arange(n_pad, device=like.device, dtype=like.dtype)[None, :]
        idx_m = torch.arange(m_pad, device=like.device, dtype=like.dtype)[None, :]
        n_i = n_valid.to(like.dtype)[:, None]
        m_i = m_valid.to(like.dtype)[:, None]
        x1 = torch.exp(self.lambda_ * -torch.square((idx_n + 1 - (n_i + 1) / 2) / ((n_i + 1) / 2)))
        x2 = torch.exp(self.lambda_ * -torch.square((idx_m + 1 - (m_i + 1) / 2) / ((m_i + 1) / 2)))
        w = x1[:, :, None] * x2[:, None, :]
        return ((1 - self.theta) * w + self.theta)[:, None]

    def _pooled_mask(self, n_valid, m_valid, pooled):
        """Cells each pair's own max-pool would produce, on the padded output."""
        k, p = self.pool_width, self.pool_width // 2
        out_n = torch.div(n_valid + 2 * p - k, k, rounding_mode="floor") + 1
        out_m = torch.div(m_valid + 2 * p - k, k, rounding_mode="floor") + 1
        n2, m2 = pooled.shape[-2:]
        rows = torch.arange(n2, device=pooled.device)[None, :] < out_n[:, None]
        cols = torch.arange(m2, device=pooled.device)[None, :] < out_m[:, None]
        return (rows[:, :, None] & cols[:, None, :])[:, None].to(pooled.dtype)

    # -- forward ---------------------------------------------------------------

    def forward(self, batch: PairBatch, tile_rows: int | None = None) -> dict[str, torch.Tensor]:
        e0, mask0 = self._embed(batch.z0, batch.f0)
        e1, mask1 = self._embed(batch.z1, batch.f1)
        pair_mask = (mask0[:, :, None] & mask1[:, None, :])[:, None]  # (b, 1, N, M)

        C = self.contact_map(e0, e1, pair_mask, tile_rows)
        pm = pair_mask.to(C.dtype)
        k_real = pm.sum(dim=(1, 2, 3)).clamp(min=1)
        cmap_magnitude = (C * pm).sum(dim=(1, 2, 3)) / k_real

        n_valid, m_valid = mask0.sum(1), mask1.sum(1)
        yhat = C * self._weight_matrix(n_valid, m_valid, C.shape[2], C.shape[3], C) if self.weighting else C

        if self.pool_width:
            pool = nn.functional.max_pool2d(
                yhat.masked_fill(pm == 0, float("-inf")), self.pool_width, padding=self.pool_width // 2
            )
            pm = self._pooled_mask(n_valid, m_valid, pool)
            yhat = pool.masked_fill(pm == 0, 0.0)
        else:
            yhat = yhat * pm

        # Mean of the cells above mu + gamma * sigma, per pair, over real cells only.
        k = pm.sum(dim=(1, 2, 3))
        mu = yhat.sum(dim=(1, 2, 3)) / k.clamp(min=1)
        sigma = (((yhat - mu.view(-1, 1, 1, 1)) * pm) ** 2).sum(dim=(1, 2, 3)) / (k - 1).clamp(min=1)
        Q = torch.relu(yhat - mu.view(-1, 1, 1, 1) - self.gamma * sigma.view(-1, 1, 1, 1)) * pm
        p_hat = Q.sum(dim=(1, 2, 3)) / (torch.sign(Q).sum(dim=(1, 2, 3)) + 1)
        if self.sigmoid:
            p_hat = torch.sigmoid(self.sigmoid_k * (p_hat - self.sigmoid_x0))
        return {"p_hat": p_hat, "cmap_magnitude": cmap_magnitude}

    # -- saving and loading -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        save_model(self, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> DScriptBatched:
        return load_into(cls, path, device)