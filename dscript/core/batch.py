"""PairBatch: the input type of the model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class PairBatch:
    """
    Pairs as ragged per-protein tensors. The serial model runs them one at a
    time; the batched model pads them.

    ``z0``/``z1`` are language-model embeddings ``(L, d)``. ``f0``/``f1`` are
    the concatenated per-residue features ``(L, F)``, with F = 0 when none are
    used. ``y`` holds labels (zeros at prediction time), ``glider`` optional
    GLIDER scores, and ``index`` each pair's position in its PairSet, used to
    put predictions from several devices back in order.
    """

    z0: list[torch.Tensor]
    z1: list[torch.Tensor]
    f0: list[torch.Tensor]
    f1: list[torch.Tensor]
    y: torch.Tensor
    glider: torch.Tensor | None
    index: torch.Tensor

    def __len__(self) -> int:
        return len(self.z0)

    @property
    def len0(self) -> list[int]:
        return [t.shape[0] for t in self.z0]

    @property
    def len1(self) -> list[int]:
        return [t.shape[0] for t in self.z1]

    def subset(self, idx: list[int]) -> PairBatch:
        return PairBatch(
            z0=[self.z0[i] for i in idx],
            z1=[self.z1[i] for i in idx],
            f0=[self.f0[i] for i in idx],
            f1=[self.f1[i] for i in idx],
            y=self.y[idx],
            glider=None if self.glider is None else self.glider[idx],
            index=self.index[idx],
        )

    def swapped(self) -> PairBatch:
        """The same pairs with the two sides exchanged."""
        return replace(self, z0=self.z1, z1=self.z0, f0=self.f1, f1=self.f0)

    def _apply(self, fn) -> PairBatch:
        return replace(
            self,
            z0=[fn(t) for t in self.z0],
            z1=[fn(t) for t in self.z1],
            f0=[fn(t) for t in self.f0],
            f1=[fn(t) for t in self.f1],
            y=fn(self.y),
            glider=None if self.glider is None else fn(self.glider),
            index=fn(self.index),
        )

    def to(self, device) -> PairBatch:
        return self._apply(lambda t: t.to(device, non_blocking=True))

    def pin_memory(self) -> PairBatch:  # called by DataLoader(pin_memory=True)
        return self._apply(lambda t: t.pin_memory())
