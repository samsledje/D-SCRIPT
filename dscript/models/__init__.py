"""
The two D-SCRIPT model families. They share inputs (PairBatch), outputs
(``{"p_hat", "cmap_magnitude"}``) and the saved-file format, but not weights.

    serial/      DScriptSerial - the original architecture  (dscript train_serial)
    batched/     DScriptBatched - batch-native, tiled       (dscript train_batched)
    saving.py    one-file model format; load_model picks the right class

The two families never import each other.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .batched import DScriptBatched
from .saving import read_header
from .serial import DScriptSerial

FAMILIES = {cls.family: cls for cls in (DScriptSerial, DScriptBatched)}


def load_model(path: str | Path, device: str | torch.device = "cpu"):
    """Load any saved D-SCRIPT model, whichever family it is."""
    family, _ = read_header(path)
    if family not in FAMILIES:
        raise ValueError(f"{path} holds an unknown model family {family!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[family].load(path, device)


__all__ = ["FAMILIES", "DScriptBatched", "DScriptSerial", "load_model"]
