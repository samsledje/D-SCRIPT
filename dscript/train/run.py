"""
Set up and run one training process: read data, build the model, train.

``run(fabric, cfg)`` is what Fabric launches on every device, for both
commands; the config's class (SerialTrainConfig or BatchedTrainConfig)
selects the model family. It is also the library entry point:

    from dscript.train import BatchedTrainConfig, run
    from dscript.core.runtime import make_fabric

    cfg = BatchedTrainConfig(train=..., val=..., embeddings=..., output_dir=...)
    make_fabric(cfg).launch(run, cfg)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from lightning_fabric import Fabric
from loguru import logger

from .. import __version__
from ..core.data import PairSet, open_embeddings, read_features, read_pairs
from ..core.runtime import configure_logging, local_processes, set_cpu_threads, set_numerics
from .config import TrainConfig
from .families import FAMILIES
from .loss import Objective
from .steps import effective_pairs_per_step, glider_scores
from .trainer import Trainer


def run(fabric: Fabric, cfg: TrainConfig) -> None:
    family = FAMILIES[cfg.family]
    out_dir = Path(cfg.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    configure_logging(fabric.global_rank, out_dir / "train.log")
    if fabric.is_global_zero:
        (out_dir / "config.yaml").write_text(cfg.to_yaml())
        logger.info(f"D-SCRIPT {__version__}; called as: {' '.join(sys.argv)}")
        logger.info(f"Training the {cfg.family} model on {fabric.world_size} process(es), {fabric.device.type}")
    set_numerics()
    set_cpu_threads(fabric, cfg.num_nodes)
    fabric.seed_everything(cfg.seed, verbose=False)  # identical initial weights everywhere

    # --- data: lengths and features read once, on rank 0 ------------------------
    train_a, train_b, train_y = read_pairs(cfg.train)
    augment = getattr(cfg, "augment", False)  # a serial-only setting
    if augment:  # every pair in both orders, as the original does
        train_a, train_b = train_a + train_b, train_b + train_a
        train_y = np.concatenate([train_y, train_y])
    val_a, val_b, val_y = read_pairs(cfg.val)
    proteins = sorted(set(train_a) | set(train_b) | set(val_a) | set(val_b))
    embeddings = open_embeddings(cfg.embeddings)
    prepared = None
    if fabric.is_global_zero:
        logger.info(f"Reading lengths of {len(proteins)} proteins from {cfg.embeddings}")
        lengths = embeddings.lengths(proteins)
        prepared = (lengths, read_features(cfg.feature_fastas, lengths))
    lengths, features = fabric.broadcast(prepared)

    trainset = PairSet(train_a, train_b, train_y, lengths, embeddings, features, longer_first=family.longer_first)
    valset = PairSet(val_a, val_b, val_y, lengths, embeddings, features, longer_first=family.longer_first)
    # pairs_per_step is a ceiling, like --batch-size in `dscript train`: a training
    # set smaller than it trains as one step of every pair. Capped here, once, so
    # the steps, the loss scaling and the step count in the logs all agree.
    capped = effective_pairs_per_step(len(trainset), cfg.pairs_per_step)
    if capped != cfg.pairs_per_step:
        if fabric.is_global_zero:
            logger.warning(
                f"pairs_per_step ({cfg.pairs_per_step}) is larger than the training set ({len(trainset)} pairs); "
                f"using {capped}"
            )
        cfg = cfg.model_copy(update={"pairs_per_step": capped})
    if cfg.topsy_turvy:
        scores = glider_scores(cfg.train, trainset.p0, trainset.p1, cfg.glider_thresh) if fabric.is_global_zero else None
        trainset.glider = torch.from_numpy(fabric.broadcast(scores))
    if fabric.is_global_zero:
        lens = np.concatenate([trainset.len0, trainset.len1])
        logger.info(
            f"{len(trainset)} training pairs, {len(valset)} validation pairs; protein length "
            f"median {np.median(lens):.0f}, max {lens.max()}; "
            f"{len(trainset) // cfg.pairs_per_step} steps of {cfg.pairs_per_step} pairs per epoch"
            + (" (training pairs include both orders)" if augment else "")
        )

    # --- model ---------------------------------------------------------------
    raw = family.model(
        input_dim=embeddings.get(proteins[0]).shape[-1],
        features={name: size for name, (_, size) in features.items()},
        **cfg.model_kwargs(),
    )
    optimizer = torch.optim.Adam([p for p in raw.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    model, optimizer = fabric.setup(raw, optimizer)
    if fabric.is_global_zero:
        logger.info(f"Model: {sum(p.numel() for p in raw.parameters()):,} parameters; config {raw.config}")

    planner = family.planner.for_device(
        fabric.device,
        cfg.memory_fraction,
        local_processes(fabric, cfg.num_nodes),
        **family.planner_args(cfg, sum(raw.features.values())),
    )
    if fabric.is_global_zero:
        i = int(np.argmax(trainset.cost))
        need = planner.estimate([int(trainset.len0[i])], [int(trainset.len1[i])], training=True)
        msg = (
            f"Largest training pair: {trainset.p0[i]} - {trainset.p1[i]} ({trainset.len0[i]} x {trainset.len1[i]}), "
            f"about {need / 2**30:.1f} GB to train; memory budget {planner.budget / 2**30:.1f} GB per device"
        )
        (logger.warning if need > planner.budget else logger.info)(msg + (" - it may not fit" if need > planner.budget else ""))

    wandb_run = None
    if cfg.wandb_project and fabric.is_global_zero:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.wandb_name,
            tags=cfg.wandb_tags, config=cfg.model_dump(mode="json"), dir=str(out_dir),
        )

    objective = Objective(cfg.interaction_weight, cfg.glider_weight if cfg.topsy_turvy else None)
    trainer = Trainer(fabric, cfg, model, optimizer, raw, planner, objective, trainset, valset, wandb_run, family.steps)
    if cfg.resume is not None:
        trainer.resume(cfg.resume)
    try:
        trainer.fit()
    finally:
        if wandb_run is not None:
            wandb_run.finish()