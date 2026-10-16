"""
Splits a device's pairs into chunks that fit in memory - for training steps
and for inference alike. One planner per model family, because the families
use memory differently:

* ``SerialMemoryPlanner`` - pairs run one at a time, unpadded. Training keeps
  every pair's activations until backward (proportional to the chunk's total
  contact-map cells) plus the backward workspace of its largest pair;
  inference needs only the largest pair's forward workspace.
* ``BatchedMemoryPlanner`` - pairs are padded to the chunk's largest N and M,
  and the contact map is computed in row tiles, so it also chooses how many
  rows go in a tile. For this family a chunk that merely fits is not good
  enough: padding and tiling both cost compute that the budget does not
  measure, so chunks are also bounded by how much padding they add and how
  many tiles they force.

Per-cell costs were measured in fp32. The starting budget comes from the free
GPU memory; an out-of-memory error lowers it for the rest of the run, so a bad
estimate costs a retried chunk, never the run. Chunking never changes results:
predictions are per pair, and the training loss is a sum over pairs.

The trainer and ``core.inference`` use only the shared interface:
``chunks``, ``forward_kwargs``, ``padded_cells``, ``shrink``, ``estimate``
and ``single_pair_retries``.
"""

from __future__ import annotations

import math
import os

import torch
from loguru import logger

SHRINK = 0.7
MIN_BUDGET_BYTES = 64 * 2**20
HEADROOM = 1.15


class OutOfMemory(RuntimeError):
    pass


class MemoryPlanner:
    """Budget bookkeeping shared by both families."""

    # How many times a chunk holding a single pair is retried, with a lower
    # budget each time, before that pair is reported as too large.
    single_pair_retries = 1

    def __init__(self, budget_bytes: float):
        self.budget = float(budget_bytes)
        self.n_shrinks = 0

    @classmethod
    def for_device(cls, device: torch.device, memory_fraction: float, local_processes: int, **kw) -> MemoryPlanner:
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            # What is already in use - CUDA context, model, optimizer, other
            # processes - comes out of this device's share.
            budget = memory_fraction * total - (total - free)
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            budget = memory_fraction * total / max(local_processes, 1) / 2
        if budget < MIN_BUDGET_BYTES:
            raise OutOfMemory(f"only {budget / 2**20:.0f} MB available on {device}")
        return cls(budget, **kw)

    def shrink(self, device: torch.device, what: str) -> None:
        """Lower the budget after an out-of-memory error."""
        self.budget *= SHRINK
        self.n_shrinks += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if self.budget < MIN_BUDGET_BYTES:
            raise OutOfMemory(f"out of memory on {device} even with a {self.budget / 2**20:.0f} MB budget ({what})")
        logger.warning(f"Out of memory on {device} ({what}); memory budget lowered to {self.budget / 2**30:.2f} GB")

    # -- per family ---------------------------------------------------------------

    def estimate(self, len0: list[int], len1: list[int], training: bool) -> float:
        """Estimated peak bytes for one chunk."""
        raise NotImplementedError

    def chunks(self, len0: list[int], len1: list[int], training: bool) -> list[list[int]]:
        """
        Group positions 0..b-1 into chunks under the budget, largest pair
        first. A pair over budget on its own still gets its own chunk; if it
        then runs out of memory, the caller retries it and finally reports it.
        """
        order = sorted(range(len(len0)), key=lambda i: self.sort_key(len0[i], len1[i]))
        out, cur = [], []
        for i in order:
            trial = cur + [i]
            if cur and not self.accepts([len0[j] for j in trial], [len1[j] for j in trial], training):
                out.append(cur)
                trial = [i]
            cur = trial
        if cur:
            out.append(cur)
        return out

    def accepts(self, len0: list[int], len1: list[int], training: bool) -> bool:
        """
        Whether these pairs may share a chunk. Memory is the only condition
        here; a family whose chunks also cost compute adds its own.
        """
        return self.estimate(len0, len1, training) <= self.budget

    def sort_key(self, n: int, m: int):
        """Order in which pairs fill chunks: largest contact map first."""
        return (-n * m, -n)

    def forward_kwargs(self, len0: list[int], len1: list[int], training: bool = True) -> dict:
        """Extra arguments for ``model(chunk, ...)``."""
        return {}

    def padded_cells(self, len0: list[int], len1: list[int]) -> int:
        """Contact-map cells actually computed for a chunk, padding included."""
        return sum(a * b for a, b in zip(len0, len1, strict=True))


class SerialMemoryPlanner(MemoryPlanner):
    """For DScriptSerial. No padding and no tiling."""

    single_pair_retries = 1

    def __init__(self, budget_bytes: float, hidden_dim: int, pair_dim: int):
        super().__init__(budget_bytes)
        d, h = pair_dim, hidden_dim  # pair_dim = projection_dim + feature widths
        # Bytes per contact-map cell, fitted to measurements.
        self.train_kept = 4 * (2.5 * d + 3.5 * h)
        self.train_work = 4 * (1.0 * d + 3.5 * h)
        self.eval_work = 4 * (3.0 * d + 3.5 * h)

    def estimate(self, len0, len1, training):
        cells = [a * b for a, b in zip(len0, len1, strict=True)]
        if not cells:
            return 0.0
        if training:
            return HEADROOM * (self.train_kept * sum(cells) + self.train_work * max(cells))
        return HEADROOM * self.eval_work * max(cells)


class BatchedMemoryPlanner(MemoryPlanner):
    """
    For DScriptBatched. A chunk is padded to its largest N and M, and its
    contact map is computed in row tiles.

    Memory splits in two. What the chunk *keeps* for the whole of backward is
    a few single-channel maps per padded cell - the pair features are not
    kept, because a tiled map recomputes them. What is left is the workspace
    of the tile being computed, which is what sets the rows per tile.

    Unlike the serial family, a bigger chunk is not simply better here:

    * padded cells are ``b * max(len0) * max(len1)``, so one long pair makes
      every other pair in the chunk compute padding it does not need;
    * the tile workspace is linear in ``b``, so a bigger chunk means fewer
      rows per tile, which means more tiles, more halo rows recomputed and
      (in training) more checkpoint recomputation.

    So ``accepts`` stops a chunk growing once it would compute more than
    ``max_padding_waste`` times the cells its pairs really need, or force the
    map into more than ``max_tiles`` tiles - whichever comes first. A single
    pair is always allowed to form a chunk on its own, so an unusually long
    protein ends up in a chunk of its own rather than padding everything
    around it.
    """

    # A single pair can always be retried with smaller tiles.
    single_pair_retries = 8

    # Bytes per padded cell kept across the whole chunk: the single-channel
    # contact map, the weighted and thresholded maps the pooling builds, and
    # their gradients. Measured in fp32.
    TRAIN_KEPT_BYTES_PER_CELL = 32
    EVAL_KEPT_BYTES_PER_CELL = 12
    # Bytes per cell per hidden channel in the tile being computed.
    TILE_BYTES_PER_CELL_PER_CHANNEL = 20  # measured about 15
    # Defaults for the two compute bounds above.
    MAX_PADDING_WASTE = 1.2
    MAX_TILES = 4

    def __init__(
        self,
        budget_bytes: float,
        hidden_dim: int,
        pair_dim: int,
        halo: int,
        max_padding_waste: float | None = None,
        max_tiles: int | None = None,
    ):
        super().__init__(budget_bytes)
        self.hidden_dim = hidden_dim
        self.pair_dim = pair_dim
        self.halo = halo
        self.max_padding_waste = self.MAX_PADDING_WASTE if max_padding_waste is None else float(max_padding_waste)
        self.max_tiles = self.MAX_TILES if max_tiles is None else int(max_tiles)

    # -- the two parts of the budget ------------------------------------------

    def _kept(self, b: int, n: int, m: int, training: bool) -> float:
        per_cell = self.TRAIN_KEPT_BYTES_PER_CELL if training else self.EVAL_KEPT_BYTES_PER_CELL
        return per_cell * b * n * m

    def _tile_bytes_per_row(self, b: int, m: int) -> float:
        return b * self.hidden_dim * (self.TILE_BYTES_PER_CELL_PER_CHANNEL * m + 8 * self.pair_dim)

    def _rows(self, b: int, n: int, m: int, training: bool) -> int:
        """Rows per tile that fit in whatever the chunk does not keep. At least 1."""
        free = self.budget / HEADROOM - self._kept(b, n, m, training)
        if free <= 0:
            return 1
        rows = int(free / self._tile_bytes_per_row(b, m)) - 2 * self.halo
        return max(1, min(rows, n))

    # -- the shared interface --------------------------------------------------

    def estimate(self, len0, len1, training):
        if not len0:
            return 0.0
        b, n, m = len(len0), max(len0), max(len1)
        kept = self._kept(b, n, m, training)
        rows = self._rows(b, n, m, training)
        return HEADROOM * (kept + self._tile_bytes_per_row(b, m) * (rows + 2 * self.halo))

    def accepts(self, len0, len1, training):
        if not super().accepts(len0, len1, training):
            return False
        b, n, m = len(len0), max(len0), max(len1)
        if math.ceil(n / self._rows(b, n, m, training)) > self.max_tiles:
            return False
        real = sum(x * y for x, y in zip(len0, len1, strict=True))
        return self.padded_cells(len0, len1) <= self.max_padding_waste * real

    def padded_cells(self, len0, len1):
        return len(len0) * max(len0, default=0) * max(len1, default=0)

    def sort_key(self, n, m):
        # Longest first, so a chunk's first pair fixes its padded N.
        return (-n, -m)

    def tile_rows(self, b: int, n: int, m: int, training: bool = True) -> int | None:
        """Rows per tile for b pairs padded to n x m; None computes the whole map at once."""
        rows = self._rows(b, n, m, training)
        return None if rows >= n else rows

    def forward_kwargs(self, len0, len1, training: bool = True):
        return {"tile_rows": self.tile_rows(len(len0), max(len0), max(len1), training)}


def pair_too_large(name0: str, name1: str, n: int, m: int, device: torch.device, planner: MemoryPlanner, training: bool) -> OutOfMemory:
    need = planner.estimate([n], [m], training) / 2**30
    return OutOfMemory(
        f"The pair {name0} ({n} residues) - {name1} ({m} residues) does not fit in memory on {device} "
        f"by itself (a {n} x {m} contact map; about {need:.1f} GB {'to train' if training else 'to score'}). "
        "Remove it from the pair file, or use a GPU with more memory."
    )