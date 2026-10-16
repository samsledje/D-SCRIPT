"""
What each optimizer step trains on: the epoch's global steps, each device's
share of them, and the GLIDER targets for Topsy-Turvy.

Two ways to form steps, one per model family:

* ``random_steps`` (serial) - a plain shuffle, as in the original. The serial
  model runs pairs one at a time with no padding, so grouping pairs by length
  would save nothing.
* ``bucketed_steps`` (batched) - pairs of similar length share a step, so
  little of each padded batch is padding.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.data import PairSet, deal
from ..glider import glide_compute_map, glider_score

# Residues per length band when grouping similar pairs into a step.
BAND = 64
# Steps pooled before sorting by length; larger pools mean less padding but
# less randomness in which pairs share a step.
POOL_STEPS = 50

Steps = Callable[[PairSet, int, int, int], list[np.ndarray]]


def effective_pairs_per_step(n_pairs: int, pairs_per_step: int) -> int:
    """``pairs_per_step`` is a ceiling: a training set smaller than it is one step."""
    if n_pairs < 1:
        raise ValueError("the training set is empty")
    return min(pairs_per_step, n_pairs)


def random_steps(pairs: PairSet, pairs_per_step: int, seed: int, epoch: int) -> list[np.ndarray]:
    """
    One epoch's global steps: a fresh shuffle, cut into steps of exactly
    ``pairs_per_step`` pairs (the remainder is dropped, differently each epoch).
    """
    pairs_per_step = effective_pairs_per_step(len(pairs), pairs_per_step)
    n = len(pairs)
    order = np.random.default_rng([seed, epoch]).permutation(n)
    return [order[i : i + pairs_per_step] for i in range(0, (n // pairs_per_step) * pairs_per_step, pairs_per_step)]


def bucketed_steps(pairs: PairSet, pairs_per_step: int, seed: int, epoch: int) -> list[np.ndarray]:
    """
    One epoch's global steps of exactly ``pairs_per_step`` pairs, each holding
    pairs of similar length, in shuffled order.
    """
    pairs_per_step = effective_pairs_per_step(len(pairs), pairs_per_step)
    n = len(pairs)
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(n)[: (n // pairs_per_step) * pairs_per_step]
    steps = []
    pool = pairs_per_step * POOL_STEPS
    for start in range(0, len(order), pool):
        chunk = order[start : start + pool]
        chunk = chunk[np.lexsort((pairs.len1[chunk], pairs.len0[chunk] // BAND))]
        steps += [chunk[i : i + pairs_per_step] for i in range(0, len(chunk), pairs_per_step)]
    return [steps[i] for i in rng.permutation(len(steps))]


def train_plan(
    pairs: PairSet, steps: Steps, pairs_per_step: int, seed: int, epoch: int, rank: int, world_size: int
) -> list[list[int]]:
    """This device's share of every step of the epoch, balanced by cost."""
    return [deal(step, pairs.cost, world_size)[rank] for step in steps(pairs, pairs_per_step, seed, epoch)]


def glider_scores(train_path: Path, p0, p1, thresh: float) -> np.ndarray:
    """GLIDER score per pair, from the training positives. Symmetric in pair order."""
    raw = pd.read_csv(train_path, sep="\t", header=None, dtype={0: str, 1: str})
    mat, gmap = glide_compute_map(raw[raw.iloc[:, 2] == 1], thres_p=thresh * 100)
    return np.array(
        [glider_score(a, b, gmap, mat) for a, b in zip(p0, p1, strict=True)],
        dtype=np.float32,
    )