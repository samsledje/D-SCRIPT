"""
Score a whole pair set across devices, memory-safely.

Each device takes a cost-balanced share of the pairs (``data.shard``) and runs
them in chunks sized by the MemoryPlanner; a chunk that runs out of memory is
split and retried on that device alone. A single pair that still does not fit
after its retries (``planner.single_pair_retries``) stops the run with an
error naming it. ``gather_scores`` then puts every
device's scores back in pair order on rank 0.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.distributed as dist

from .data import PairSet, loader, shard
from .memory import MemoryPlanner, pair_too_large

OOM = torch.cuda.OutOfMemoryError


@torch.no_grad()
def score_pairs(
    model,
    pairs: PairSet,
    planner: MemoryPlanner,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    workers: int = 0,
    group: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This device's share of the predictions.

    :param model: called as ``model(batch, **planner.forward_kwargs(...))``, returning ``{"p_hat": ...}``
    :return: pair indices and their predicted interaction probabilities
    """
    was_training = model.training
    model.eval()
    index, scores = [], []
    for batch in loader(pairs, shard(pairs, group, rank, world_size), workers, device.type == "cuda"):
        queue = deque((idx, 0) for idx in planner.chunks(batch.len0, batch.len1, training=False))
        while queue:
            idx, attempts = queue.popleft()
            p_hat, failed = None, False
            try:
                chunk = batch.subset(idx).to(device)
                p_hat = model(chunk, **planner.forward_kwargs(chunk.len0, chunk.len1, training=False))["p_hat"].float().cpu()
            except OOM:
                failed = True
            chunk = None  # outside the except block, so the failed attempt is freed
            if failed:
                if len(idx) == 1 and attempts >= planner.single_pair_retries:
                    i = int(batch.index[idx[0]])
                    raise pair_too_large(
                        pairs.p0[i], pairs.p1[i], int(pairs.len0[i]), int(pairs.len1[i]), device, planner, training=False
                    )
                planner.shrink(device, f"inference chunk of {len(idx)} pairs")
                half = len(idx) // 2
                queue.extendleft([(idx[half:], 0), (idx[:half], 0)] if len(idx) > 1 else [(idx, attempts + 1)])
                continue
            index.append(batch.index[idx])
            scores.append(p_hat)
    model.train(was_training)
    if not index:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    return torch.cat(index).numpy(), torch.cat(scores).numpy()


def gather_scores(index: np.ndarray, scores: np.ndarray, n: int, world_size: int, rank: int) -> np.ndarray | None:
    """Every device's scores, in pair order, on rank 0; None elsewhere. A collective."""
    parts = [(index, scores)]
    if world_size > 1:
        parts = [None] * world_size
        dist.all_gather_object(parts, (index, scores))
    if rank != 0:
        return None
    out = np.full(n, np.nan, dtype=np.float32)
    for idx, s in parts:
        out[idx] = s
    if np.isnan(out).any():
        raise RuntimeError(f"{int(np.isnan(out).sum())} pairs were not scored")
    return out