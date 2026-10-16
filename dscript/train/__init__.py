"""
Training both D-SCRIPT model families. The ``dscript train_serial`` and
``dscript train_batched`` commands are thin wrappers around ``run``.

    config.py      TrainConfig (shared) and SerialTrainConfig, BatchedTrainConfig
    families.py    everything that differs between the two families, in one table
    run.py         run(fabric, cfg): read data, build the model, train
    steps.py       random or length-bucketed steps, per-device shares, GLIDER targets
    loss.py        the objective, summed over pairs
    trainer.py     the loop: memory-safe steps, validation, checkpoints

    example_serial.yaml, example_batched.yaml   every setting with its default
"""

from .config import BatchedTrainConfig, SerialTrainConfig, TrainConfig
from .run import run

__all__ = ["BatchedTrainConfig", "SerialTrainConfig", "TrainConfig", "run"]
