"""
The serial D-SCRIPT model: the original architecture, trained by ``dscript train_serial``.

Projection (embedding.py) -> pair features and contact map (contact.py) ->
weighted, thresholded pooling (here). Pairs run one at a time, exactly as in
the original: in training, the BatchNorms take their statistics from the
single pair being processed.

Saved models are single files; see ``models/saving.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ...core.batch import PairBatch
from ..saving import load_into, save_model
from .contact import ContactCNN
from .embedding import FullyConnectedEmbed


class DScriptSerial(nn.Module):
    """
    ``forward(batch) -> {"p_hat": (b,), "cmap_magnitude": (b,)}``. Pairs run one at a time.

    :param input_dim: language-model embedding dimension
    :param features: per-residue features appended after projection,
        ``{name: width}`` in the order the batch concatenates them
    """

    family = "serial"

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
        self.embedding = FullyConnectedEmbed(input_dim, projection_dim, dropout)
        self.contact = ContactCNN(projection_dim + sum(features.values()), hidden_dim, kernel_width)

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
        self.contact.clip()
        if self.weighting:
            self.theta.data.clamp_(0, 1)
            self.lambda_.data.clamp_(min=0)
        self.gamma.data.clamp_(min=0)

    # -- one pair ----------------------------------------------------------------

    def contact_map(self, z0, z1, f0, f1) -> torch.Tensor:
        """``(N, d0), (M, d0)`` embeddings and ``(N, F), (M, F)`` features -> ``(1, 1, N, M)``."""
        e0 = torch.cat([self.embedding(z0.unsqueeze(0)), f0.unsqueeze(0)], dim=2)
        e1 = torch.cat([self.embedding(z1.unsqueeze(0)), f1.unsqueeze(0)], dim=2)
        return self.contact(e0, e1)

    def _weight_matrix(self, n: int, m: int, like: torch.Tensor) -> torch.Tensor:
        xx_n = torch.arange(n, device=like.device, dtype=like.dtype)
        xx_m = torch.arange(m, device=like.device, dtype=like.dtype)
        x1 = torch.exp(self.lambda_ * -torch.square((xx_n + 1 - (n + 1) / 2) / ((n + 1) / 2)))
        x2 = torch.exp(self.lambda_ * -torch.square((xx_m + 1 - (m + 1) / 2) / ((m + 1) / 2)))
        return (1 - self.theta) * (x1.unsqueeze(1) * x2) + self.theta

    def interaction(self, C: torch.Tensor) -> torch.Tensor:
        """Pool one contact map to an interaction probability: the mean of cells above mu + gamma * sigma."""
        yhat = C * self._weight_matrix(C.shape[2], C.shape[3], C) if self.weighting else C
        if self.pool_width:
            yhat = nn.functional.max_pool2d(yhat, self.pool_width, padding=self.pool_width // 2)
        mu = torch.mean(yhat)
        sigma = torch.var(yhat)
        Q = torch.relu(yhat - mu - self.gamma * sigma)
        p_hat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        if self.sigmoid:
            p_hat = torch.sigmoid(self.sigmoid_k * (p_hat - self.sigmoid_x0))
        return p_hat.reshape(())

    # -- a batch -------------------------------------------------------------------

    def forward(self, batch: PairBatch) -> dict[str, torch.Tensor]:
        p_hat, magnitude = [], []
        for z0, z1, f0, f1 in zip(batch.z0, batch.z1, batch.f0, batch.f1, strict=True):
            C = self.contact_map(z0, z1, f0, f1)
            p_hat.append(self.interaction(C))
            magnitude.append(torch.mean(C))
        return {"p_hat": torch.stack(p_hat), "cmap_magnitude": torch.stack(magnitude)}

    # -- saving and loading -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        save_model(self, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> DScriptSerial:
        return load_into(cls, path, device)
