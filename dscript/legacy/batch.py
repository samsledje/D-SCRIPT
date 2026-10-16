"""
The single input type for training-time forward passes.  # new (step 1)

A :class:`PairBatch` holds a minibatch of protein pairs as ragged lists - one
``(L, d)`` tensor per protein - plus a dictionary of optional per-residue
features (Foldseek, backbone, ...). A single pair is simply a batch of one.

Keeping the batch ragged means one type serves both model families:

* v1 (:class:`~dscript.legacy.interaction.ModelInteraction`) iterates over the
  pairs, exactly reproducing the original per-pair training;
* v2 (:class:`the new model`) packs and
  pads them with :meth:`PairBatch.padded` / :meth:`PairBatch.packed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

# Order in which per-residue features are concatenated onto the projected
# embedding. Matches the order ModelInteraction.cpred used (foldseek, then
# backbone), so checkpoints trained before the refactor stay valid.
FEATURE_ORDER: tuple[str, ...] = ("foldseek", "backbone")


def ordered_feature_names(names) -> list[str]:  # new (step 1)
    """Known features in FEATURE_ORDER first, then any others alphabetically."""
    names = set(names)
    known = [n for n in FEATURE_ORDER if n in names]
    return known + sorted(names - set(FEATURE_ORDER))


@dataclass
class PairBatch:  # new (step 1)
    """
    A minibatch of protein pairs.

    :param z0: first protein of each pair, each ``(L_i, d)``
    :param z1: second protein of each pair, each ``(M_i, d)``
    :param features: ``{name: (list for z0 side, list for z1 side)}``, each
        tensor ``(L_i, f)``
    :param y: labels ``(b,)``, or None at inference
    :param names0: protein names of the first side
    :param names1: protein names of the second side
    """

    z0: list[torch.Tensor]
    z1: list[torch.Tensor]
    features: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = field(
        default_factory=dict
    )
    y: torch.Tensor | None = None
    names0: list[str] = field(default_factory=list)
    names1: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.z0) != len(self.z1):
            raise ValueError(f"z0 has {len(self.z0)} proteins but z1 has {len(self.z1)}")
        for name, (f0, f1) in self.features.items():
            if len(f0) != len(self.z0) or len(f1) != len(self.z1):
                raise ValueError(f"feature {name!r} does not match the batch size")
            for i, (a, b) in enumerate(zip(f0, f1, strict=True)):
                if a.shape[0] != self.z0[i].shape[0] or b.shape[0] != self.z1[i].shape[0]:
                    raise ValueError(
                        f"feature {name!r} length differs from its embedding at pair {i}"
                    )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pair(cls, z0, z1, features=None, y=None) -> PairBatch:
        """A batch of one pair. Accepts ``(L, d)`` or ``(1, L, d)`` tensors."""

        def _2d(t):
            return t.squeeze(0) if t.dim() == 3 else t

        feats = {k: ([_2d(a)], [_2d(b)]) for k, (a, b) in (features or {}).items()}
        return cls([_2d(z0)], [_2d(z1)], feats, y)

    # -- basic properties -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.z0)

    @property
    def feature_names(self) -> list[str]:
        return ordered_feature_names(self.features)

    @property
    def lengths0(self) -> torch.Tensor:
        return torch.tensor([t.shape[0] for t in self.z0], dtype=torch.long)

    @property
    def lengths1(self) -> torch.Tensor:
        return torch.tensor([t.shape[0] for t in self.z1], dtype=torch.long)

    def pair(self, i: int) -> PairBatch:
        """The i-th pair as a batch of one."""
        feats = {k: ([f0[i]], [f1[i]]) for k, (f0, f1) in self.features.items()}
        y = None if self.y is None else self.y[i : i + 1]
        n0 = self.names0[i : i + 1] if self.names0 else []
        n1 = self.names1[i : i + 1] if self.names1 else []
        return PairBatch([self.z0[i]], [self.z1[i]], feats, y, n0, n1)

    # -- device movement ------------------------------------------------------

    def _apply(self, fn) -> PairBatch:
        return replace(
            self,
            z0=[fn(t) for t in self.z0],
            z1=[fn(t) for t in self.z1],
            features={
                k: ([fn(t) for t in f0], [fn(t) for t in f1])
                for k, (f0, f1) in self.features.items()
            },
            y=None if self.y is None else fn(self.y),
        )

    def to(self, device, non_blocking: bool = True) -> PairBatch:
        return self._apply(lambda t: t.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> PairBatch:
        # DataLoader(pin_memory=True) calls this on custom batch types.
        return self._apply(lambda t: t.pin_memory())

    # -- layouts for batched models -------------------------------------------

    def side(self, which: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        One side of the batch with its features concatenated on the last dim:
        (embeddings, features-or-empty) as parallel lists.
        """
        zs = self.z0 if which == 0 else self.z1
        names = self.feature_names
        feats = []
        for i in range(len(zs)):
            parts = [self.features[n][which][i] for n in names]
            feats.append(torch.cat(parts, dim=-1) if parts else zs[i].new_zeros(zs[i].shape[0], 0))
        return zs, feats

    @staticmethod
    def pad(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:  # new (step 3)
        """Pad ``(L_i, d)`` tensors to ``(b, L_max, d)`` plus a ``(b, L_max)`` mask."""
        b = len(seqs)
        lmax = max(s.shape[0] for s in seqs)
        out = seqs[0].new_zeros((b, lmax) + tuple(seqs[0].shape[1:]))
        mask = torch.zeros((b, lmax), dtype=torch.bool, device=seqs[0].device)
        for i, s in enumerate(seqs):
            out[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = True
        return out, mask

    @staticmethod
    def pack(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:  # new (step 3)
        """
        Concatenate ``(L_i, d)`` tensors into one ``(sum L_i, d)`` tensor plus
        cumulative offsets ``(b + 1,)`` - the layout variable-length attention
        kernels call ``cu_seqlens``.
        """
        lengths = torch.tensor([s.shape[0] for s in seqs], device=seqs[0].device)
        offsets = torch.zeros(len(seqs) + 1, dtype=torch.long, device=seqs[0].device)
        offsets[1:] = torch.cumsum(lengths, 0)
        return torch.cat(seqs, dim=0), offsets
