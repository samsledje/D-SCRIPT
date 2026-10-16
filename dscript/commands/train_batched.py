"""
``dscript train_batched``: train the batched D-SCRIPT model (batch-native,
tiled contact maps).

    dscript train_batched --config run.yaml                      # all visible GPUs (or CPU)
    dscript train_batched --config run.yaml --devices 2 --lr 3e-4
    dscript train_batched --config run.yaml --accelerator cpu --devices 4
    torchrun --nproc-per-node 4 -m dscript.commands.train_batched --config run.yaml --devices 4
    srun ... dscript train_batched --config run.yaml --devices 4 --num-nodes 2   # SLURM

Every setting is listed in ``dscript/train/example_batched.yaml`` and ``dscript train_batched --help``.
"""

from __future__ import annotations

import sys

from ..core.runtime import make_fabric
from ..train import BatchedTrainConfig, run


def main(argv: list[str] | None = None) -> None:
    cfg = BatchedTrainConfig.parse(sys.argv[1:] if argv is None else argv, prog="dscript train_batched")
    make_fabric(cfg).launch(run, cfg)


if __name__ == "__main__":
    main()
