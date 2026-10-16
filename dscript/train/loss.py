"""
The D-SCRIPT objective, as a sum over pairs.

    interaction = glider_weight * BCE(p_hat, glider) + (1 - glider_weight) * BCE(p_hat, y)
                  (BCE(p_hat, y) alone without Topsy-Turvy)
    loss        = lambda * interaction + (1 - lambda) * contact magnitude

Every term is summed, not averaged. The trainer divides by the fixed number of
pairs in the global step, so each pair carries the same weight however the
step was split across devices and memory-sized chunks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.batch import PairBatch


class Objective:
    def __init__(self, interaction_weight: float, glider_weight: float | None = None):
        self.lam = interaction_weight
        self.glider_weight = glider_weight

    def __call__(self, out: dict[str, torch.Tensor], batch: PairBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p = out["p_hat"]
        bce = F.binary_cross_entropy(p, batch.y, reduction="sum")
        if self.glider_weight is not None:
            glider = F.binary_cross_entropy(p, batch.glider, reduction="sum")
            interaction = self.glider_weight * glider + (1 - self.glider_weight) * bce
        else:
            interaction = bce
        magnitude = out["cmap_magnitude"].sum()
        loss = self.lam * interaction + (1 - self.lam) * magnitude
        with torch.no_grad():
            sums = {
                "loss": loss.detach(),
                "interaction_loss": interaction.detach(),
                "magnitude_loss": magnitude.detach(),
                "correct": ((p > 0.5).float() == batch.y).sum(),
                "sq_err": ((p - batch.y) ** 2).sum(),
                "n": torch.tensor(float(len(batch)), device=p.device),
            }
        return loss, sums
