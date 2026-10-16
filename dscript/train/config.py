"""
Settings for ``dscript train_serial`` and ``dscript train_batched``.

``TrainConfig`` holds everything the two share; each command's class adds or
changes only what differs. Hardware settings come from RuntimeSettings.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from ..core.runtime import RuntimeSettings


class TrainConfig(RuntimeSettings):
    """Settings shared by both training commands."""

    family: ClassVar[str]  # which model family this command trains

    # --- data ----------------------------------------------------------------
    train: Path = Field(description="training pairs: prot1 <tab> prot2 <tab> label")
    val: Path = Field(description="validation pairs, same format")
    embeddings: Path = Field(description="HDF5 file, or a directory of per-protein .pt files")
    foldseek_fasta: Path | None = Field(None, description="Foldseek 3Di sequences; enables that feature")
    backbone_fasta: Path | None = Field(None, description="12-state backbone sequences; enables that feature")

    # --- model -------------------------------------------------------------------
    projection_dim: int = Field(100, ge=1, description="projected embedding dimension")
    hidden_dim: int = Field(50, ge=1, description="pair-feature channels")
    kernel_width: int = Field(7, ge=1, description="contact conv width; must be odd")
    dropout: float = Field(0.5, ge=0.0, lt=1.0, description="dropout on the projection")
    weighting: bool = Field(True, description="apply the learned distance-weighting matrix W")
    sigmoid: bool = Field(True, description="final logistic activation on the pooled score")
    pool_width: int | None = Field(None, ge=1, description="local max-pool width before global pooling; off if unset")
    theta_init: float = Field(1.0, description="initial theta of W")
    lambda_init: float = Field(0.0, description="initial lambda of W")
    gamma_init: float = Field(0.0, description="initial gamma of the pooling threshold")
    sigmoid_k: float = Field(20.0, description="slope of the final logistic activation")
    sigmoid_x0: float = Field(0.5, description="midpoint of the final logistic activation")

    # --- loss --------------------------------------------------------------------
    interaction_weight: float = Field(0.35, ge=0.0, le=1.0, description="lambda: weight on the interaction loss vs contact magnitude")
    topsy_turvy: bool = Field(False, description="add the GLIDER (Topsy-Turvy) loss")
    glider_weight: float = Field(0.2, ge=0.0, le=1.0, description="weight of GLIDER within the interaction loss")
    glider_thresh: float = Field(0.925, gt=0.0, lt=1.0, description="GLIDER score percentile treated as an edge")

    # --- optimisation ------------------------------------------------------------
    epochs: int = Field(10, ge=1)
    lr: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)

    # --- output ------------------------------------------------------------------
    output_dir: Path = Field(description="run directory: config, log, checkpoints")
    resume: Path | None = Field(None, description="resume from a last.ckpt written by an earlier run")
    report_every: int = Field(50, ge=1, description="steps between progress reports")
    wandb_project: str | None = Field(None, description="log to Weights & Biases when set")
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: list[str] | None = Field(None, description="comma-separated on the command line")

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if self.kernel_width % 2 == 0:
            raise ValueError("kernel_width must be odd (an even width shifts the contact map by one cell)")
        return self

    @property
    def feature_fastas(self) -> dict[str, Path]:
        """Enabled per-residue features, in the order they are concatenated."""
        out = {}
        if self.foldseek_fasta is not None:
            out["foldseek"] = self.foldseek_fasta
        if self.backbone_fasta is not None:
            out["backbone"] = self.backbone_fasta
        return out

    def model_kwargs(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "projection_dim", "hidden_dim", "kernel_width", "dropout", "weighting", "sigmoid", "pool_width",
                "theta_init", "lambda_init", "gamma_init", "sigmoid_k", "sigmoid_x0",
            )
        }


class SerialTrainConfig(TrainConfig):
    """``dscript train_serial``: the original model, trained as the original is."""

    family: ClassVar[str] = "serial"

    pairs_per_step: int = Field(25, ge=1, description="pairs per optimizer step, across all devices")
    augment: bool = Field(True, description="also train on every pair in reverse order (B, A), as the original does")


class BatchedTrainConfig(TrainConfig):
    """``dscript train_batched``: the batch-native model."""

    family: ClassVar[str] = "batched"

    pairs_per_step: int = Field(32, ge=1, description="pairs per optimizer step, across all devices")
    max_padding_waste: float = Field(
        1.15, ge=1.0, description="cells a memory chunk may compute, as a multiple of the cells its pairs need"
    )
    max_tiles: int = Field(4, ge=1, description="most row tiles a chunk's contact map may be split into")