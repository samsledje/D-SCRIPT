"""
Data: pair files, lazily read embeddings, per-residue features, pair sets,
sharding across devices, and loaders.

Embeddings are never loaded wholesale. Each process (and each DataLoader
worker) opens the file itself and reads only the proteins in the batch it is
building, so the operating system's page cache holds one shared copy however
many GPUs and workers there are.

Pairs keep the order they have in the pair file, unless ``longer_first`` is
set: the batched model gives the same prediction for (A, B) and (B, A), and
putting the longer protein on the rows lets its contact map be tiled along the
long side.
"""

from __future__ import annotations

import heapq
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger

from ..fasta import parse_dict
from ..foldseek import build_backbone_vocab, fold_vocab
from .batch import PairBatch

FEATURE_VOCABS = {"foldseek": fold_vocab, "backbone": build_backbone_vocab()}


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


def read_pairs(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path, sep="\t", header=None, dtype={0: str, 1: str})
    if df.shape[1] < 3:
        raise ValueError(f"{path}: expected 3 tab-separated columns (prot1, prot2, label)")
    return list(df[0]), list(df[1]), df[2].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(0) if x.dim() == 3 else x


class H5Embeddings:
    """One dataset per protein, (L, d) or (1, L, d). Reopened in each process."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._file = None
        self._pid = None

    @property
    def file(self) -> h5py.File:
        # h5py handles do not survive fork; reopen in every new process.
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.__init__(state["path"])

    def lengths(self, names) -> dict[str, int]:
        f = self.file
        missing = [n for n in names if n not in f]
        if missing:
            raise KeyError(f"{len(missing)} proteins not in {self.path}, e.g. {missing[:5]}")
        return {n: int(f[n].shape[-2]) for n in names}

    def get(self, name: str) -> torch.Tensor:
        return _as_2d(torch.from_numpy(self.file[name][()])).float()


class PtDirEmbeddings:
    """One ``.pt`` tensor per protein, at ``<dir>/<a>/<ab>/<name>.pt`` or ``<dir>/<name>.pt``."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _file(self, name: str) -> Path:
        nested = self.path / name[:1] / name[:2] / f"{name}.pt"
        return nested if nested.exists() else self.path / f"{name}.pt"

    def lengths(self, names) -> dict[str, int]:
        missing = [n for n in names if not self._file(n).exists()]
        if missing:
            raise KeyError(f"{len(missing)} proteins have no .pt file in {self.path}, e.g. {missing[:5]}")
        return {n: self._length(n) for n in names}

    def _length(self, name: str) -> int:
        try:  # mmap reads only the header, not the tensor
            t = torch.load(self._file(name), map_location="cpu", mmap=True, weights_only=True)
        except RuntimeError:  # files saved in PyTorch's legacy format cannot be mapped
            t = torch.load(self._file(name), map_location="cpu", weights_only=True)
        return int(_as_2d(t).shape[0])

    def get(self, name: str) -> torch.Tensor:
        return _as_2d(torch.load(self._file(name), map_location="cpu", weights_only=True)).float()


def open_embeddings(path: Path) -> H5Embeddings | PtDirEmbeddings:
    path = Path(path)
    if path.is_dir():
        return PtDirEmbeddings(path)
    if path.is_file() and h5py.is_hdf5(path):
        return H5Embeddings(path)
    raise FileNotFoundError(f"{path} is neither an HDF5 file nor a directory of .pt files")


# ---------------------------------------------------------------------------
# Per-residue features
# ---------------------------------------------------------------------------


def read_feature(fasta: Path, vocab: dict[str, int], lengths: dict[str, int]) -> tuple[dict[str, np.ndarray], int]:
    """
    Token indices per protein (int8, -1 = no feature, which one-hot encodes as
    all zeros). Proteins absent from the FASTA, or whose sequence length does
    not match their embedding, get no feature - as in the original training.

    :return: indices by protein, and how many proteins got no feature
    """
    seqs = parse_dict(str(fasta))
    out, n_missing = {}, 0
    for name, length in lengths.items():
        seq = seqs.get(name)
        if seq is None or len(seq) != length:
            out[name] = np.full(length, -1, dtype=np.int8)
            n_missing += 1
            continue
        try:
            out[name] = np.fromiter((vocab[c] for c in seq), dtype=np.int8, count=length)
        except KeyError as e:
            raise ValueError(f"{fasta}: protein {name} has a letter outside the vocabulary: {e}") from None
    return out, n_missing


def read_features(fastas: dict[str, Path], lengths: dict[str, int]) -> dict[str, tuple[dict[str, np.ndarray], int]]:
    """
    Every enabled feature, as ``{feature: (indices by protein, vocabulary size)}``,
    in the order given - which is the order the model concatenates them.
    """
    features = {}
    for name, fasta in fastas.items():
        if name not in FEATURE_VOCABS:
            raise ValueError(f"unknown feature {name!r}; known: {sorted(FEATURE_VOCABS)}")
        idx, n_missing = read_feature(fasta, FEATURE_VOCABS[name], lengths)
        if n_missing:
            logger.warning(f"{n_missing} proteins have no {name} feature (absent from {fasta} or wrong length); using zeros")
        features[name] = (idx, len(FEATURE_VOCABS[name]))
    return features


def one_hot(idx: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(idx.astype(np.int64))
    out = torch.zeros(len(idx), size)
    keep = t >= 0
    out[keep, t[keep]] = 1.0
    return out


# ---------------------------------------------------------------------------
# Pair sets
# ---------------------------------------------------------------------------


class PairSet:
    """
    A pair file with everything needed to build batches from it. Picklable,
    so DataLoader workers get their own lazily opened embedding handle.
    """

    def __init__(self, p0, p1, y, lengths: dict[str, int], embeddings, features: dict, glider=None, longer_first=False):
        """``y`` may be None when there are no labels (prediction)."""
        if longer_first:
            swap = [lengths[a] < lengths[b] for a, b in zip(p0, p1, strict=True)]
            self.p0 = [b if s else a for a, b, s in zip(p0, p1, swap, strict=True)]
            self.p1 = [a if s else b for a, b, s in zip(p0, p1, swap, strict=True)]
        else:
            self.p0, self.p1 = list(p0), list(p1)
        self.y = torch.zeros(len(p0)) if y is None else torch.as_tensor(y, dtype=torch.float32)
        self.len0 = np.array([lengths[a] for a in self.p0], dtype=np.int64)
        self.len1 = np.array([lengths[b] for b in self.p1], dtype=np.int64)
        self.embeddings = embeddings
        self.features = features  # {name: (indices by protein, vocab size)}
        self.glider = None if glider is None else torch.as_tensor(glider, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.p0)

    @property
    def cost(self) -> np.ndarray:
        """Contact-map cells per pair - what memory and compute scale with."""
        return self.len0 * self.len1

    def _features(self, protein: str, length: int) -> torch.Tensor:
        parts = [one_hot(by_protein[protein], size) for by_protein, size in self.features.values()]
        return torch.cat(parts, dim=-1) if parts else torch.zeros(length, 0)

    def collate(self, idx: list[int]) -> PairBatch:
        idx = list(idx)
        z0 = [self.embeddings.get(self.p0[i]) for i in idx]
        z1 = [self.embeddings.get(self.p1[i]) for i in idx]
        return PairBatch(
            z0=z0,
            z1=z1,
            f0=[self._features(self.p0[i], len(z)) for i, z in zip(idx, z0, strict=True)],
            f1=[self._features(self.p1[i], len(z)) for i, z in zip(idx, z1, strict=True)],
            y=self.y[idx],
            glider=None if self.glider is None else self.glider[idx],
            index=torch.tensor(idx, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Sharding and loading
# ---------------------------------------------------------------------------


def deal(indices, cost: np.ndarray, world_size: int) -> list[list[int]]:
    """
    Split pairs across devices so each gets about the same total cost:
    largest first, each to the least-loaded device. Deterministic, so every
    device computes the same split. With at least as many pairs as devices,
    every device gets at least one.
    """
    heap = [(0, r) for r in range(world_size)]
    out: list[list[int]] = [[] for _ in range(world_size)]
    for i in sorted(indices, key=lambda i: -int(cost[i])):
        load, r = heapq.heappop(heap)
        out[r].append(int(i))
        heapq.heappush(heap, (load + int(cost[i]), r))
    return out


def shard(pairs: PairSet, group: int, rank: int, world_size: int) -> list[list[int]]:
    """This device's share of every pair, in groups of similar length - for scoring a whole set."""
    mine = deal(range(len(pairs)), pairs.cost, world_size)[rank]
    mine.sort(key=lambda i: (pairs.len0[i], pairs.len1[i]))
    return [mine[i : i + group] for i in range(0, len(mine), group)]


class PlanDataset(torch.utils.data.Dataset):
    """Item i is the batch for the plan's i-th group of pairs."""

    def __init__(self, pairs: PairSet, plan: list[list[int]]):
        self.pairs = pairs
        self.plan = plan

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int) -> PairBatch:
        return self.pairs.collate(self.plan[i])


def loader(pairs: PairSet, plan: list[list[int]], workers: int, pin: bool) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        PlanDataset(pairs, plan),
        batch_size=None,  # items are already batches
        shuffle=False,  # the plan fixes the order
        num_workers=workers,
        pin_memory=pin,
        prefetch_factor=2 if workers else None,
    )
