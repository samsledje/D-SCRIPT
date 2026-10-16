"""
``dscript train_serial``: train the serial D-SCRIPT model (the original
architecture; pairs run one at a time).

    dscript train_serial --config run.yaml                      # all visible GPUs (or CPU)
    dscript train_serial --config run.yaml --devices 2 --lr 3e-4
    dscript train_serial --config run.yaml --accelerator cpu --devices 4
    torchrun --nproc-per-node 4 -m dscript.commands.train_serial --config run.yaml --devices 4
    srun ... dscript train_serial --config run.yaml --devices 4 --num-nodes 2   # SLURM

Every setting is listed in ``dscript/train/example_serial.yaml`` and ``dscript train_serial --help``.
"""

from __future__ import annotations

import sys

from ..core.runtime import make_fabric
from ..train import SerialTrainConfig, run


def main(argv: list[str] | None = None) -> None:
    cfg = SerialTrainConfig.parse(sys.argv[1:] if argv is None else argv, prog="dscript train_serial")
    make_fabric(cfg).launch(run, cfg)


if __name__ == "__main__":
    main()
