"""
The training loop.

A step is ``pairs_per_step`` pairs across all devices. Each device runs its
share in memory-sized chunks with gradient sync switched off, accumulating
``loss_sum / pairs_per_step``; then the gradients are summed across devices
once, and the optimizer steps. Because nothing is communicated until every
chunk has finished, a chunk that runs out of memory is retried in smaller
pieces on that device alone - the other devices just wait a little longer.

The trainer is the same for both model families; how a family uses memory
lives in its MemoryPlanner, and how its steps are formed in ``steps``.

BatchNorm running statistics (serial model only; the batched model has none)
are updated pair by pair on each device, as in the original, and averaged
across devices once per step, so every device continues from - and saves -
the same values. A chunk that fails is rolled back completely, statistics
included, before it is retried.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from ..core.batch import PairBatch
from ..core.data import PairSet, loader
from ..core.inference import gather_scores, score_pairs
from ..core.memory import MemoryPlanner, pair_too_large
from ..core.metrics import interaction_metrics
from .config import TrainConfig
from .loss import Objective
from .steps import Steps, random_steps, train_plan

OOM = torch.cuda.OutOfMemoryError
# Running totals per report window. A fixed list: every device must send the
# same values in the same order to the all-reduce.
STAT_KEYS = (
    "n", "loss", "interaction_loss", "magnitude_loss", "correct", "sq_err",
    "cells", "padded_cells", "oom_retries", "data_s", "compute_s", "steps",
)


class Trainer:
    def __init__(self, fabric, cfg: TrainConfig, model, optimizer, raw_model, planner: MemoryPlanner,
                 objective: Objective, trainset: PairSet, valset: PairSet, run=None, steps: Steps = random_steps):
        self.fabric = fabric
        self.steps = steps  # how the epoch's steps are formed (random or bucketed by length)
        self.cfg = cfg
        self.model = model  # Fabric-wrapped, for training
        self.raw = raw_model  # unwrapped: evaluation, clip(), saving
        self.optimizer = optimizer
        self.planner = planner
        self.objective = objective
        self.trainset = trainset
        self.valset = valset
        self.run = run
        self.device = fabric.device
        self.params = [p for p in raw_model.parameters() if p.requires_grad]
        # BatchNorm running statistics; num_batches_tracked is unused with momentum.
        self.buffers = [b for b in raw_model.buffers() if b.dtype.is_floating_point]
        self.ckpt_dir = Path(cfg.output_dir) / "checkpoints"
        self.start_epoch = 0
        self.best_aupr = -math.inf
        self.fault = None  # tests set this to inject an out-of-memory error

    # ------------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------------

    def _zero_grads(self) -> None:
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            else:
                p.grad.zero_()

    def train_step(self, batch: PairBatch, stats: dict) -> None:
        self._zero_grads()
        queue = deque((idx, 0) for idx in self.planner.chunks(batch.len0, batch.len1, training=True))
        while queue:
            idx, attempts = queue.popleft()
            grads = [p.grad.clone() for p in self.params]
            running = [b.clone() for b in self.buffers]
            chunk = out = loss = None
            failed = False
            try:
                chunk = batch.subset(idx).to(self.device)
                if self.fault is not None:
                    self.fault(chunk)
                with self.fabric.no_backward_sync(self.model):
                    out = self.model(chunk, **self.planner.forward_kwargs(chunk.len0, chunk.len1, training=True))
                    loss, sums = self.objective(out, chunk)
                    self.fabric.backward(loss / self.cfg.pairs_per_step)
            except OOM:
                failed = True
            if failed:
                # Outside the except block, so the failed attempt's tensors are
                # released before retrying. Undo everything the attempt changed.
                chunk = out = loss = None
                for p, g in zip(self.params, grads, strict=True):
                    p.grad.copy_(g)
                for b, r in zip(self.buffers, running, strict=True):
                    b.copy_(r)
                if len(idx) == 1 and attempts >= self.planner.single_pair_retries:
                    i = int(batch.index[idx[0]])
                    ts = self.trainset
                    raise pair_too_large(ts.p0[i], ts.p1[i], int(ts.len0[i]), int(ts.len1[i]), self.device, self.planner, training=True)
                self.planner.shrink(self.device, f"chunk of {len(idx)} pairs")
                half = len(idx) // 2
                queue.extendleft([(idx[half:], 0), (idx[:half], 0)] if len(idx) > 1 else [(idx, attempts + 1)])
                stats["oom_retries"] += 1
                continue
            for k, v in sums.items():
                stats[k] += float(v)
            stats["cells"] += sum(a * b for a, b in zip(chunk.len0, chunk.len1, strict=True))
            stats["padded_cells"] += self.planner.padded_cells(chunk.len0, chunk.len1)
            del chunk, out, loss

        self._sync_gradients()
        self.optimizer.step()
        self.raw.clip()
        self._sync_running_stats()

    def _sync_gradients(self) -> None:
        """Sum gradients across devices: one collective per step."""
        if self.fabric.world_size == 1:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in self.params])
        flat = self.fabric.all_reduce(flat, reduce_op="sum")
        offset = 0
        for p in self.params:
            p.grad.copy_(flat[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    def _sync_running_stats(self) -> None:
        """Average BatchNorm running statistics across devices: one collective per step."""
        if self.fabric.world_size == 1 or not self.buffers:
            return
        flat = torch.cat([b.reshape(-1) for b in self.buffers])
        flat = self.fabric.all_reduce(flat, reduce_op="mean")
        offset = 0
        for b in self.buffers:
            b.copy_(flat[offset : offset + b.numel()].view_as(b))
            offset += b.numel()

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """Each device scores its share of the validation set; every rank gets the metrics."""
        f = self.fabric
        index, scores = score_pairs(
            self.raw, self.valset, self.planner, self.device, f.global_rank, f.world_size,
            self.cfg.workers, group=self.cfg.pairs_per_step,
        )
        p_hat = gather_scores(index, scores, len(self.valset), f.world_size, f.global_rank)
        metrics = interaction_metrics(torch.from_numpy(p_hat), self.valset.y) if f.is_global_zero else None
        return f.broadcast(metrics)

    # ------------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------------

    def fit(self) -> None:
        cfg, fabric = self.cfg, self.fabric
        n_steps = len(self.trainset) // cfg.pairs_per_step
        for epoch in range(self.start_epoch, cfg.epochs):
            # Dropout differs per device and per epoch, and a resumed run
            # continues with the same randomness it would have had.
            torch.manual_seed(int(np.random.SeedSequence([cfg.seed, epoch, fabric.global_rank]).generate_state(1)[0]))
            plan = train_plan(
                self.trainset, self.steps, cfg.pairs_per_step, cfg.seed, epoch, fabric.global_rank, fabric.world_size
            )
            self.model.train()
            window = defaultdict(float)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            t_prev = time.perf_counter()
            for step, batch in enumerate(loader(self.trainset, plan, cfg.workers, self.device.type == "cuda")):
                t_data = time.perf_counter()
                window["data_s"] += t_data - t_prev
                self.train_step(batch, window)
                t_prev = time.perf_counter()
                window["compute_s"] += t_prev - t_data
                window["steps"] += 1
                if (step + 1) % cfg.report_every == 0 or step + 1 == n_steps:
                    self._report(epoch, step + 1, n_steps, window)
                    window = defaultdict(float)

            metrics = self.evaluate()
            if fabric.is_global_zero:
                logger.info(
                    f"Epoch {epoch + 1}/{cfg.epochs} validation: "
                    + ", ".join(f"{k}={v:.6g}" for k, v in metrics.items() if k != "n")
                )
                if self.run is not None:
                    self.run.log({"epoch": epoch + 1, **{f"val/{k}": v for k, v in metrics.items()}})
            self._save(epoch, metrics)

    def _report(self, epoch: int, step: int, n_steps: int, w: dict) -> None:
        local = torch.tensor([w[k] for k in STAT_KEYS], dtype=torch.float64, device=self.device)
        g = dict(zip(STAT_KEYS, self.fabric.all_reduce(local, reduce_op="sum").tolist(), strict=True))
        if not self.fabric.is_global_zero or g["n"] == 0:
            return
        n = g["n"]
        row = {
            "train/loss": g["loss"] / n,
            "train/interaction_loss": g["interaction_loss"] / n,
            "train/magnitude_loss": g["magnitude_loss"] / n,
            "train/accuracy": g["correct"] / n,
            "train/mse": g["sq_err"] / n,
            "speed/cells_per_s": g["cells"] / max(g["compute_s"], 1e-9) * self.fabric.world_size,
            "speed/padding_efficiency": g["cells"] / max(g["padded_cells"], 1),
            "speed/data_ms": 1000 * g["data_s"] / max(g["steps"], 1),
            "speed/compute_ms": 1000 * g["compute_s"] / max(g["steps"], 1),
            "memory/oom_retries": g["oom_retries"],
            "memory/budget_gb": self.planner.budget / 2**30,
        }
        line = (
            f"[{epoch + 1}/{self.cfg.epochs}] step {step}/{n_steps}: loss={row['train/loss']:.5f} "
            f"interaction={row['train/interaction_loss']:.5f} magnitude={row['train/magnitude_loss']:.5f} "
            f"accuracy={row['train/accuracy']:.2%} | {row['speed/cells_per_s'] / 1e6:.1f}M cells/s, "
            f"padding {row['speed/padding_efficiency']:.0%}, "
            f"per step data {row['speed/data_ms']:.0f} ms, compute {row['speed/compute_ms']:.0f} ms"
        )
        if self.device.type == "cuda":
            row["memory/peak_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
            line += f" | GPU peak {row['memory/peak_gb']:.1f} GB of {self.planner.budget / 2**30:.1f} GB budget"
            torch.cuda.reset_peak_memory_stats(self.device)
        if g["oom_retries"]:
            line += f" | {g['oom_retries']:.0f} out-of-memory retries"
        logger.info(line)
        if self.run is not None:
            self.run.log(row)

    # ------------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------------

    def _save(self, epoch: int, metrics: dict) -> None:
        """
        ``last.ckpt`` holds everything needed to resume. Weights are also
        saved as model files (``DScript.load``) - per epoch, ``best`` by
        validation AUPR, and ``final``.
        """
        is_best = metrics["aupr"] > self.best_aupr
        if is_best:
            self.best_aupr = metrics["aupr"]
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "epoch": epoch,
            "best_aupr": self.best_aupr,
            "memory_budget": self.planner.budget,
        }
        self.fabric.save(self.ckpt_dir / "last.ckpt", state)
        if self.fabric.is_global_zero:
            names = [f"epoch{epoch + 1:03d}"] + (["best"] if is_best else []) + (["final"] if epoch + 1 == self.cfg.epochs else [])
            for name in names:
                self.raw.save(self.ckpt_dir / f"{name}.safetensors")
            if is_best:
                logger.info(f"New best validation AUPR {self.best_aupr:.6g} at epoch {epoch + 1}")
        self.fabric.barrier()

    def resume(self, path: Path) -> None:
        state = {"model": self.model, "optimizer": self.optimizer}
        rest = self.fabric.load(path, state)
        self.start_epoch = rest["epoch"] + 1
        self.best_aupr = rest["best_aupr"]
        self.planner.budget = min(self.planner.budget, rest["memory_budget"])
        logger.info(f"Resumed from {path}: continuing at epoch {self.start_epoch + 1}")