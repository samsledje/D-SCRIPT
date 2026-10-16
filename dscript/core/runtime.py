"""
Running on hardware: the settings every command shares, process launch with
Lightning Fabric, logging, and numerics.

Launching: a command calls ``make_fabric(settings).launch(fn, ...)``. Fabric
starts one process per device itself, or uses the processes started by
torchrun or SLURM when it detects them.
"""

from __future__ import annotations

import inspect
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Literal

import torch
from lightning_fabric import Fabric
from lightning_fabric.strategies import DDPStrategy
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from pydantic import Field

from .settings import Settings


class RuntimeSettings(Settings):
    accelerator: Literal["auto", "cpu", "gpu"] = Field("auto", description="auto uses GPUs when present")
    devices: int | Literal["auto"] = Field("auto", description="processes per node: GPUs, or CPU processes with --accelerator cpu")
    num_nodes: int = Field(1, ge=1)
    workers: int = Field(4, ge=0, description="data-loading worker processes per device")
    memory_fraction: float = Field(0.85, gt=0.1, le=0.98, description="share of each GPU's memory to plan for")
    collective_timeout_minutes: float = Field(30.0, gt=0.0, description="how long a device waits for the others before failing")
    seed: int = 0


def _no_forward_buffer_sync() -> dict:
    """
    DDP copies rank 0's buffers (BatchNorm running statistics) to every device
    at each forward pass. That is a collective inside the forward pass, which
    would make devices that retry a chunk after running out of memory fall
    out of step; training averages the statistics once per step instead.
    The option was renamed in recent PyTorch releases.
    """
    params = inspect.signature(DistributedDataParallel.__init__).parameters
    return {"forward_sync_buffers": False} if "forward_sync_buffers" in params else {"broadcast_buffers": False}


def make_fabric(settings: RuntimeSettings) -> Fabric:
    # Variable-size batches fragment PyTorch's GPU memory cache; expandable
    # segments avoid most of the resulting out-of-memory errors. Must be set
    # before CUDA starts; launched processes inherit it.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return Fabric(
        accelerator=settings.accelerator,
        devices=settings.devices,
        num_nodes=settings.num_nodes,
        # DDP's own gradient and buffer sync are never used: training syncs
        # both by hand once per step, so out-of-memory recovery stays local.
        strategy=DDPStrategy(timeout=timedelta(minutes=settings.collective_timeout_minutes), **_no_forward_buffer_sync()),
        precision="32-true",
    )


def set_numerics() -> None:
    """fp32 storage; TF32 matmuls and convolutions on tensor-core GPUs."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True


def configure_logging(rank: int, log_file: Path | None = None) -> None:
    """Rank 0 logs everything; other ranks only warnings and errors."""
    logger.remove()
    if rank == 0:
        logger.add(sys.stdout, level="INFO")
        if log_file is not None:
            logger.add(log_file, level="INFO")
    else:
        logger.add(sys.stderr, level="WARNING", format=f"rank {rank} | {{time:HH:mm:ss}} | {{level}} | {{message}}")


def local_processes(fabric: Fabric, num_nodes: int) -> int:
    return max(1, fabric.world_size // num_nodes)


def set_cpu_threads(fabric: Fabric, num_nodes: int) -> None:
    """With several CPU processes on one machine, split the cores between them."""
    if fabric.device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // local_processes(fabric, num_nodes)))
