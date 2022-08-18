"""
Train a new model.
"""
from __future__ import annotations
import argparse
import datetime
import gzip as gz
import logging as lg
import os
import subprocess as sp
import sys

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
from typing import Callable, NamedTuple, Optional

from ..datamodules import PPIDataModule

# from ..models.contact import ContactCNN
# from ..models.embedding import FullyConnectedEmbed, IdentityEmbed
# from ..models.interaction import ModelInteraction
from ..models.lightning import LitInteraction
from ..utils import config_logger


class TrainArguments(NamedTuple):
    cmd: str
    device: int
    train: str
    test: str
    embedding: str
    no_augment: bool
    input_dim: int
    projection_dim: int
    dropout: float
    hidden_dim: int
    kernel_width: int
    no_w: bool
    no_sigmoid: bool
    do_pool: bool
    pool_width: int
    num_epochs: int
    batch_size: int
    weight_decay: float
    lr: float
    interaction_weight: float
    run_tt: bool
    glider_weight: float
    glider_thresh: float
    outfile: Optional[str]
    save_prefix: Optional[str]
    checkpoint: Optional[str]
    func: Callable[[TrainArguments], None]


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """

    data_grp = parser.add_argument_group("Data")
    proj_grp = parser.add_argument_group("Projection Module")
    contact_grp = parser.add_argument_group("Contact Module")
    inter_grp = parser.add_argument_group("Interaction Module")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")

    # Data
    data_grp.add_argument("--train", help="Training data", required=True)
    data_grp.add_argument("--val", help="Validation data", required=True)
    data_grp.add_argument("--test", help="Testing data")
    data_grp.add_argument(
        "--embedding",
        required=True,
        help="h5py path containing embedded sequences",
    )
    data_grp.add_argument(
        "--no-augment",
        action="store_true",
        help="data is automatically augmented by adding (B A) for all pairs (A B). Set this flag to not augment data",
    )
    data_grp.add_argument(
        "--preload", action="store_true", help="Preload embeddings into memory"
    )
    # data_grp.add_argument(
    #     "--val_split",
    #     default=0.1,
    #     help="Proportion of data to use for validation",
    # )

    # Embedding model
    proj_grp.add_argument(
        "--input-dim",
        type=int,
        default=6165,
        help="dimension of input language model embedding (per amino acid) (default: 6165)",
    )
    proj_grp.add_argument(
        "--projection-dim",
        type=int,
        default=100,
        help="dimension of embedding projection layer (default: 100)",
    )
    proj_grp.add_argument(
        "--dropout-p",
        type=float,
        default=0.5,
        help="parameter p for embedding dropout layer (default: 0.5)",
    )

    # Contact model
    contact_grp.add_argument(
        "--hidden-dim",
        type=int,
        default=50,
        help="number of hidden units for comparison layer in contact prediction (default: 50)",
    )
    contact_grp.add_argument(
        "--kernel-width",
        type=int,
        default=7,
        help="width of convolutional filter for contact prediction (default: 7)",
    )

    # Interaction Model
    inter_grp.add_argument(
        "--no-w",
        action="store_true",
        help="don't use weight matrix in interaction prediction model",
    )
    inter_grp.add_argument(
        "--no-sigmoid",
        action="store_true",
        help="don't use sigmoid activation at end of interaction model",
    )
    inter_grp.add_argument(
        "--do-pool",
        action="store_true",
        help="use max pool layer in interaction prediction model",
    )
    inter_grp.add_argument(
        "--pool-width",
        type=int,
        default=9,
        help="size of max-pool in interaction model (default: 9)",
    )

    # Training
    train_grp.add_argument(
        "--epoch-scale",
        type=int,
        default=1,
        help="Report heldout performance every this many epochs (default: 1)",
    )
    train_grp.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )

    train_grp.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Minibatch size (default: 25)",
    )
    train_grp.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="L2 regularization (default: 0)",
    )
    train_grp.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    train_grp.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.35,
        help="weight on the similarity objective (default: 0.35)",
    )

    # Output
    misc_grp.add_argument(
        "-o", "--outfile", help="Output file path (default: stdout)"
    )
    misc_grp.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="Verbosity level (default: 2 [info])",
    )
    misc_grp.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    misc_grp.add_argument(
        "--save-prefix", help="Path prefix for saving models"
    )
    misc_grp.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    misc_grp.add_argument(
        "--checkpoint", help="Checkpoint model to start training from"
    )

    return parser


def main(args):
    """
    Run training from arguments.

    :meta private:
    """
    args.wandb = True

    if args.debug:
        args.verbosity = 3
        args.wandb = False

    logg = config_logger(
        args.outfile,
        "%(asctime)s [%(levelname)s] %(message)s",
        args.verbosity,
        use_stdout=True,
    )
    # logg.info(f"Beginning experiment {conf.experiment_id}")

    if args.debug:
        logg.warning("RUNNING IN DEBUG MODE")

    if args.test is None:
        args.test = args.val

    logg.info("Data:")
    logg.info(f"\ttrain file: {args.train}")
    logg.info(f"\tval file: {args.val}")
    logg.info(f"\ttest file: {args.test}")
    logg.info(f"\tdata_augmentation: {not args.no_augment}")
    logg.info(f"\tbatch_size: {args.batch_size}")
    datamod = PPIDataModule(
        args.embedding,
        args.train,
        args.val,
        args.test,
        batch_size=args.batch_size,
        preload=args.preload,
        shuffle=True,
        num_workers=0,
        augment_train=(not args.no_augment),
    )

    logg.info("Preparing data")
    datamod.prepare_data()
    logg.info("Running DataModule set up")
    datamod.setup()

    logg.info("Configuring model")
    logg.info("Initializing embedding model with:")
    logg.info(f"\tprojection_dim: {args.projection_dim}")
    logg.info(f"\tdropout_p: {args.dropout_p}")

    logg.info("Initializing contact model with:")
    logg.info(f"\thidden_dim: {args.hidden_dim}")
    logg.info(f"\tkernel_width: {args.kernel_width}")

    logg.info("Initializing interaction model with:")
    logg.info(f"\tpool_width: {args.pool_width}")
    logg.info(f"\tinteraction weight: {args.lambda_}")
    logg.info(f"\tcontact map weight: {1 - args.lambda_}")

    model = LitInteraction(
        projection_dim=args.projection_dim,
        dropout_p=args.dropout_p,
        hidden_dim=args.hidden_dim,
        kernel_width=args.kernel_width,
        pool_width=args.pool_width,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_similarity=args.lambda_,
        save_prefix=args.save_prefix,
        save_every=args.epoch_scale,
    )

    logger_list = [
        # pl_loggers.TensorBoardLogger(".", name=conf.experiment_id, default_hp_metric=False),
        pl_loggers.CSVLogger(args.save_prefix, name=args.outfile)
    ]
    # if conf.wandb:
    #     logg.info(f"Logging to WandB {conf.experiment_id}")
    #     wandb_lg = pl.loggers.WandbLogger(conf.experiment_id,
    #                                       save_dir=conf.log_dir,
    #                                       project=conf.wandb_proj,
    #                                      )
    #     logger_list.append(wandb_lg)

    logg.info(f"Saving checkpoints to '{args.save_prefix}'")
    logg.info(
        f"Training with Adam: lr={args.lr}, weight_decay={args.weight_decay}"
    )
    logg.info(f"\tnum_epochs: {args.num_epochs}")
    logg.info(f"\tepoch_scale: {args.epoch_scale}")

    num_gpus = 1 if torch.cuda.is_available else 0
    trainer = pl.Trainer(
        logger=logger_list,
        max_epochs=args.num_epochs,
        gpus=num_gpus,
    )
    trainer.fit(model, datamod)
    trainer.test(model, datamod)

    output = args.outfile
    if output is None:
        output = sys.stdout
    else:
        output = open(output, "w")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
