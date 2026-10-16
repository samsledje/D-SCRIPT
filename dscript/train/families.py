"""
What differs between the two training commands, in one place. Everything
else - data, loop, loss, checkpoints, validation - is shared.

    serial    DScriptSerial   pairs one at a time   file order        random steps
    batched   DScriptBatched  padded, tiled         longer protein    steps grouped
                                                    first             by length
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.memory import BatchedMemoryPlanner, MemoryPlanner, SerialMemoryPlanner
from ..models import DScriptBatched, DScriptSerial
from .config import TrainConfig
from .steps import Steps, bucketed_steps, random_steps


@dataclass(frozen=True)
class Family:
    model: type
    planner: type[MemoryPlanner]
    planner_args: Callable[[TrainConfig, int], dict]  # (config, feature width) -> planner arguments
    longer_first: bool  # store each pair with its longer protein first
    steps: Steps  # how an epoch's steps are formed


def _pair_args(cfg: TrainConfig, feature_dim: int) -> dict:
    return {"hidden_dim": cfg.hidden_dim, "pair_dim": cfg.projection_dim + feature_dim}


FAMILIES = {
    "serial": Family(
        model=DScriptSerial,
        planner=SerialMemoryPlanner,
        planner_args=_pair_args,
        longer_first=False,
        steps=random_steps,
    ),
    "batched": Family(
        model=DScriptBatched,
        planner=BatchedMemoryPlanner,
        planner_args=lambda cfg, f: {
            **_pair_args(cfg, f),
            "halo": cfg.kernel_width // 2,
            "max_padding_waste": cfg.max_padding_waste,
            "max_tiles": cfg.max_tiles,
        },
        longer_first=True,
        steps=bucketed_steps,
    ),
}