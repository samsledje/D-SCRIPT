"""
Model files. A saved model is one ``.safetensors`` file: its weights (and any
BatchNorm running statistics), with the model family and its constructor
arguments in the file's metadata - enough to rebuild it without any other file.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FAMILY_KEY = "dscript_family"
CONFIG_KEY = "dscript_config"


def save_model(model: torch.nn.Module, path: str | Path) -> None:
    weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(path), metadata={FAMILY_KEY: model.family, CONFIG_KEY: json.dumps(model.config)})


def read_header(path: str | Path) -> tuple[str, dict]:
    """The family (``serial`` or ``batched``) and constructor arguments stored in a model file."""
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    if FAMILY_KEY not in meta or CONFIG_KEY not in meta:
        raise ValueError(f"{path} is not a D-SCRIPT model file (no {FAMILY_KEY!r} / {CONFIG_KEY!r} metadata)")
    return meta[FAMILY_KEY], json.loads(meta[CONFIG_KEY])


def load_into(cls, path: str | Path, device: str | torch.device = "cpu"):
    """Build ``cls`` from a model file, checking the file holds that family."""
    family, config = read_header(path)
    if family != cls.family:
        raise ValueError(
            f"{path} holds a {family} model, not a {cls.family} one. "
            "Load it with its own class, or with dscript.models.load_model, which picks the class for you."
        )
    model = cls(**config)
    model.load_state_dict(load_file(str(path), device=str(device)))
    return model.to(device)
