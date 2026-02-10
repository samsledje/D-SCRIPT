"""
Train a new model.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import average_precision_score as average_precision
from torch.autograd import Variable
from tqdm import tqdm

from .. import __version__
from ..fasta import parse_dict
from ..foldseek import (
    Foldseek3diContext,
    build_backbone_vocab,
    fold_vocab,
    get_foldseek_onehot,
)
from ..glider import glide_compute_map, glider_score
from ..models.contact import ContactCNN
from ..models.embedding import FullyConnectedEmbed
from ..models.interaction_ns import InteractionInputs, ModelInteraction
from ..parallel_embedding_loader import EmbeddingLoader, add_batch_dim_if_needed
from ..utils import (
    PairedDataset,
    collate_paired_sequences,
    log,
)


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
    outfile: str | None
    save_prefix: str | None
    checkpoint: str | None
    seed: int | None
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
    foldseek_grp = parser.add_argument_group("Foldseek related commands")

    # Data
    data_grp.add_argument("--train", required=True, help="list of training pairs")
    data_grp.add_argument(
        "--test", required=True, help="list of validation/testing pairs"
    )
    # Embedding Directory
    data_grp.add_argument(
        "--embedding",
        required=True,
        help="directory containing per-protein `.pt` embeddings or HDF5 file with embeddings",
    )
    data_grp.add_argument(
        "--no-augment",
        action="store_true",
        help="data is automatically augmented by adding (B A) for all pairs (A B). Set this flag to not augment data",
    )

    # Embedding model
    proj_grp.add_argument(
        "--input-dim",
        type=int,
        default=1280,
        help="dimension of input language model embedding (per amino acid) (default: 1280), ESM-2 650M: 1280;ESM-C 600M: 1152",
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
        help="no use of weight matrix in interaction prediction model",
    )
    inter_grp.add_argument(
        "--no-sigmoid",
        action="store_true",
        help="no use of sigmoid activation at end of interaction model",
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
        "--num-epochs",
        type=int,
        default=10,
        help="number of epochs (default: 10)",
    )

    train_grp.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="minibatch size (default: 25)",
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
        help="learning rate (default: 0.001)",
    )
    train_grp.add_argument(
        "--lambda",
        dest="interaction_weight",
        type=float,
        default=0.35,
        help="weight on the similarity objective (default: 0.35)",
    )

    # Topsy-Turvy
    train_grp.add_argument(
        "--topsy-turvy",
        dest="run_tt",
        action="store_true",
        help="run in Topsy-Turvy mode -- use top-down GLIDER scoring to guide training",
    )
    train_grp.add_argument(
        "--glider-weight",
        dest="glider_weight",
        type=float,
        default=0.2,
        help="weight on the GLIDER accuracy objective (default: 0.2)",
    )
    train_grp.add_argument(
        "--glider-thresh",
        dest="glider_thresh",
        type=float,
        default=0.925,
        help="threshold beyond which GLIDER scores treated as positive edges (0 < gt < 1) (default: 0.925)",
    )

    # Output
    misc_grp.add_argument("-o", "--outfile", help="output file path (default: stdout)")
    misc_grp.add_argument("--save-prefix", help="path prefix for saving models")
    misc_grp.add_argument(
        "-d", "--device", type=int, default=-1, help="compute device to use"
    )
    misc_grp.add_argument(
        "--checkpoint", help="checkpoint model to start training from"
    )
    misc_grp.add_argument("--seed", help="Set random seed", type=int)
    misc_grp.add_argument(
        "--log_wandb", action="store_true", help="Log metrics to Weights and Biases"
    )
    misc_grp.add_argument(
        "--wandb-entity", default=None, help="Weights and Biases entity name"
    )
    misc_grp.add_argument(
        "--wandb-project", default=None, help="Weights and Biases project name"
    )

    ## Foldseek arguments
    foldseek_grp.add_argument(
        "--allow_foldseek",
        default=False,
        action="store_true",
        help="If set to true, adds the foldseek one-hot representation",
    )
    foldseek_grp.add_argument(
        "--foldseek_fasta",
        help="foldseek fasta file containing the foldseek representation",
    )
    foldseek_grp.add_argument(
        "--allow_backbone3di",
        default=False,
        action="store_true",
        help="If set to true, adds the 12 state one-hot representation",
    )
    foldseek_grp.add_argument(
        "--backbone3di_fasta",
        help="FASTA file containing the 12 state representation",
    )

    return parser


def predict_cmap_interaction(
    model,
    n0,
    n1,
    tensors,
    use_cuda,
    ### Foldseek added here
    structural_context=None,
    ###
):
    """
    Predict whether a list of protein pairs will interact, as well as their contact map.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param n0: First protein names
    :type n0: list[str]
    :param n1: Second protein names
    :type n1: list[str]
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool
    """

    b = len(n0)

    p_hat = []
    aug_x_list = []
    c_map_mag = []
    for i in range(b):
        z_a = tensors[n0[i]]  # 1 x seqlen x dim
        z_b = tensors[n1[i]]

        # Ensure 3D [B, L, D]
        z_a = add_batch_dim_if_needed(z_a)
        z_b = add_batch_dim_if_needed(z_b)

        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()

        # foldseek and backbone vectors
        f_a = f_b = b_a = b_b = None

        # TODO: make better
        if structural_context.allow_foldseek:
            assert (
                structural_context.fold_record is not None
                and structural_context.fold_vocab is not None
            )
            f_a = get_foldseek_onehot(
                n0[i],
                z_a.shape[1],
                structural_context.fold_record,
                structural_context.fold_vocab,
            ).unsqueeze(
                0
            )  # seqlen x vocabsize
            f_b = get_foldseek_onehot(
                n1[i],
                z_b.shape[1],
                structural_context.fold_record,
                structural_context.fold_vocab,
            ).unsqueeze(0)

            ## check if cuda
            if use_cuda:
                f_a = f_a.cuda()
                f_b = f_b.cuda()

        if structural_context.allow_backbone3di:
            assert (
                structural_context.backbone_record is not None
                and structural_context.fold_vocab is not None
            )
            b_a = get_foldseek_onehot(
                n0[i],
                z_a.shape[1],
                structural_context.backbone_record,
                structural_context.backbone_vocab,
            ).unsqueeze(
                0
            )  # seqlen x vocabsize
            b_b = get_foldseek_onehot(
                n1[i],
                z_b.shape[1],
                structural_context.backbone_record,
                structural_context.backbone_vocab,
            ).unsqueeze(0)

            ## check if cuda
            if use_cuda:
                b_a = b_a.cuda()
                b_b = b_b.cuda()

        cm, ph, aug_x,lam, index = model.map_predict(
            InteractionInputs(
                z_a,
                z_b,
                embed_foldseek=structural_context.allow_foldseek,
                f0=f_a,
                f1=f_b,
                embed_backbone=structural_context.allow_backbone3di,
                b0=b_a,
                b1=b_b,
            )
        )
        p_hat.append(ph)
        c_map_mag.append(torch.mean(cm))
        aug_x_list.append(aug_x.detach().cpu())
    p_hat = torch.stack(p_hat, 0).view(-1)  # [B]
    c_map_mag = torch.stack(c_map_mag, dim=0).view(-1)  # [B]
    all_aug_x = torch.cat(aug_x_list, dim=0)
    return c_map_mag, p_hat, all_aug_x,lam, index


# TODO: Remove methods??
def predict_interaction(
    model,
    n0,
    n1,
    tensors,
    use_cuda,
    ### Foldseek added here
    structural_context=None,
    ###
):
    """
    Predict whether a list of protein pairs will interact.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param n0: First protein names
    :type n0: list[str]
    :param n1: Second protein names
    :type n1: list[str]
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool
    """
    _, p_hat, _,lam, index = predict_cmap_interaction(
        model, n0, n1, tensors, use_cuda, structural_context
    )
    return p_hat


def make_mixup_params(batch_size: int, alpha: float, device):
    """
    Returns:
      perm: [B] LongTensor, random permutation indices
      lam:  [B] FloatTensor, mixing weights in [0,1]
    """
    perm = torch.randperm(batch_size, device=device)

    if alpha is None or alpha <= 0 or batch_size <= 1:
        lam = torch.ones(batch_size, device=device)
    else:
        lam = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)

    return perm, lam


import torch

import torch.nn.functional as F


def cosine_proto_pull(z_mix, y_mix, pos_proto, neg_proto, neg_weight=0.1, eps=1e-8):
    z_n = F.normalize(z_mix, p=2, dim=1, eps=eps)
    pos_n = F.normalize(pos_proto, p=2, dim=0, eps=eps)
    neg_n = F.normalize(neg_proto, p=2, dim=0, eps=eps)

    d_pos = 1.0 - (z_n * pos_n[None, :]).sum(dim=1)
    d_neg = 1.0 - (z_n * neg_n[None, :]).sum(dim=1)

    w = y_mix.clamp(0.0, 1.0)
    return (w * d_pos + neg_weight * (1.0 - w) * d_neg).mean()


import torch


@torch.no_grad()
def ema_update_protos(
    model,
    z_mix: torch.Tensor,
    y_mix: torch.Tensor,
    ema: float = 0.99,
    min_mass: float = 1e-3,
    eps: float = 1e-6,
):
    device = z_mix.device
    dtype = z_mix.dtype

    # NEVER modify y_mix in-place
    w = y_mix.to(device=device, dtype=dtype).clamp(0.0, 1.0)  # no clamp_

    wp = w.sum()
    wn = (1.0 - w).sum()

    min_mass_t = torch.tensor(min_mass, device=device, dtype=dtype)

    # ensure prototypes exist and are on the right device/dtype WITHOUT .data
    if model.pos_proto_vec.device != device or model.pos_proto_vec.dtype != dtype:
        model.pos_proto_vec = model.pos_proto_vec.to(device=device, dtype=dtype)
    if model.neg_proto_vec.device != device or model.neg_proto_vec.dtype != dtype:
        model.neg_proto_vec = model.neg_proto_vec.to(device=device, dtype=dtype)

    if wp > min_mass_t:
        wp_safe = wp.clamp_min(eps)
        batch_pos = (w[:, None] * z_mix).sum(dim=0) / wp_safe
        model.pos_proto_vec.mul_(ema).add_(batch_pos, alpha=(1.0 - ema))

    if wn > min_mass_t:
        wn_safe = wn.clamp_min(eps)
        batch_neg = ((1.0 - w)[:, None] * z_mix).sum(dim=0) / wn_safe
        model.neg_proto_vec.mul_(ema).add_(batch_neg, alpha=(1.0 - ema))


def smooth_labels(labels, smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing


def interaction_grad(
    model,
    n0,
    n1,
    y,
    tensors,
    accuracy_weight=0.35,
    run_tt=False,
    glider_weight=0,
    glider_map=None,
    glider_mat=None,
    use_cuda=True,
    ### Foldseek added here
    structural_context=None,
    # ---- prototype pull knobs
    proto_weight=0,
    proto_ema=0.99,
    proto_neg_weight=1,
):
    """
    Compute gradient and backpropagate loss for a batch.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param n0: First protein names
    :type n0: list[str]
    :param n1: Second protein names
    :type n1: list[str]
    :param y: Interaction labels
    :type y: torch.Tensor
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param accuracy_weight: Weight on the accuracy objective. Representation loss is :math:`1 - \\text{accuracy_weight}`.
    :type accuracy_weight: float
    :param run_tt: Use GLIDE top-down supervision
    :type run_tt: bool
    :param glider_weight: Weight on the GLIDE objective loss. Accuracy loss is :math:`(\\text{GLIDER_BCE}*\\text{glider_weight}) + (\\text{D-SCRIPT_BCE}*(1-\\text{glider_weight}))`.
    :type glider_weight: float
    :param glider_map: Map from protein identifier to index
    :type glider_map: dict[str, int]
    :param glider_mat: Matrix with pairwise GLIDE scores
    :type glider_mat: np.ndarray
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool

    :return: (Loss, number correct, mean square error, batch size)
    :rtype: (torch.Tensor, int, torch.Tensor, int)
    """

    c_map_mag, p_hat, z_mix,lam, index = predict_cmap_interaction(
        model, n0, n1, tensors, use_cuda, structural_context
    )
    b = len(n0)
    z_mix = z_mix.to(p_hat.device)
    if use_cuda:
        y = y.cuda()
    y = Variable(y).float().view(-1)

    # --- smooth labels
    y_mix = lam.squeeze(1) * y + (1 - lam.squeeze(1)) * y[index]
    y_mix = smooth_labels(y_mix, smoothing=0.1)

    # --- BCE (make sure shapes match)
    logits = p_hat.view(-1).float()  # rename it logits everywhere
    bce_loss = F.binary_cross_entropy_with_logits(logits, y_mix) 

    if run_tt:
        g_score = []
        for i in range(len(n0)):
            g_score.append(
                torch.tensor(
                    glider_score(n0[i], n1[i], glider_map, glider_mat),
                    dtype=torch.float64,
                )
            )
        g_score = torch.stack(g_score, 0)
        if use_cuda:
            g_score = g_score.cuda()

        glider_loss = F.binary_cross_entropy(p_hat.float(), g_score.float())
        accuracy_loss = (glider_weight * glider_loss) + ((1 - glider_weight) * bce_loss)
    else:
        accuracy_loss = bce_loss

    representation_loss = torch.mean(c_map_mag)
    
    # Reconfigure the loss
    representation_loss = representation_loss.detach() * 0.0 
    accuracy_weight = 1.0
    
    # --- prototype pull on map vectors
    proto_pull_loss = torch.tensor(0.0, device=p_hat.device)
    if proto_weight > 0:
        # lazy init prototype buffers
        D = z_mix.shape[1]
        if not hasattr(model, "pos_proto_vec"):
            model.register_buffer("pos_proto_vec", torch.zeros(D, device=p_hat.device))
            model.register_buffer("neg_proto_vec", torch.zeros(D, device=p_hat.device))

        # cosine pull loss
        proto_pull_loss = cosine_proto_pull(
            z_mix=z_mix,
            y_mix=y_mix,
            pos_proto=model.pos_proto_vec,
            neg_proto=model.neg_proto_vec,
            neg_weight=proto_neg_weight,
        )

    # --- total loss
    loss = (
        (accuracy_weight * accuracy_loss)
        + ((1.0 - accuracy_weight) * representation_loss)
        + (proto_weight * proto_pull_loss)
    )

    # Backprop Loss
    loss.backward()

    # EMA update (no grad)
    #ema_update_protos(model, z_mix.detach(), y_mix.detach(), ema=proto_ema)

    with torch.no_grad():
        p_prob = torch.sigmoid(logits)
        p_guess = (p_prob > 0.5).float()
        correct = (p_guess == y).sum().item()
        mse = ((y - p_prob) ** 2).mean().item()
        assert torch.isfinite(logits).all()

    return float(loss.item()), correct, mse, b


def interaction_eval(
    model,
    test_iterator,
    tensors,
    use_cuda,
    ### Foldseek added here
    structural_context=None,
    ###
):
    """
    Evaluate test data set performance.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param test_iterator: Test data iterator
    :type test_iterator: torch.utils.data.DataLoader
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool

    :return: (Loss, number correct, mean square error, precision, recall, F1 Score, AUPR)
    :rtype: (torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    p_hat_list = []
    y_list = []

    for n0, n1, y in test_iterator:
        # predict_interaction should return logits shaped [B] or [B,1]
        logits = predict_interaction(model, n0, n1, tensors, use_cuda, structural_context)
        p_hat_list.append(logits.view(-1))   # force [B]
        y_list.append(y.view(-1))            # force [B]

    logits = torch.cat(p_hat_list, dim=0)    # [N]
    y = torch.cat(y_list, dim=0).float()     # [N]

    device = logits.device
    y = y.to(device)

    # --- Loss: logits + targets
    loss = F.binary_cross_entropy_with_logits(logits, y).item()

    with torch.no_grad():
        p = torch.sigmoid(logits)            # [N] probabilities

        pred = (p > 0.5).float()
        correct = (pred == y).sum().item()

        # MSE should be on probabilities vs labels (not logits)
        mse = torch.mean((y - p) ** 2).item()

        tp = torch.sum(pred * y).item()
        fp = torch.sum(pred * (1 - y)).item()
        fn = torch.sum((1 - pred) * y).item()

        pr = tp / (tp + fp + 1e-8)
        re = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * re / (pr + re + 1e-8)

    # --- AUPR: use probs (recommended)
    y_np = y.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    aupr = average_precision(y_np, p_np)

    return loss, correct, mse, pr, re, f1, aupr


def train_model(args, output):
    if args.log_wandb:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=args.wandb_entity,
            # Set the wandb project where this run will be logged.
            project=args.wandb_project,
            # Track hyperparameters and run metadata.
            config=vars(args),
        )

    # Create data sets
    batch_size = args.batch_size
    use_cuda = (args.device > -1) and torch.cuda.is_available()
    train_fi = args.train
    test_fi = args.test
    no_augment = args.no_augment

    emb_path = Path(args.embedding)

    if emb_path.is_dir():
        embedding_mode = "pt_dir"
        log(f"Embedding path is a directory: {emb_path}")
    elif emb_path.is_file():
        # Could be HDF5 or something else
        if h5py.is_hdf5(str(emb_path)):
            embedding_mode = "hdf5"
            log(f"Embedding path is an HDF5 file: {emb_path}")
        else:
            raise ValueError(
                f"Embedding file is not HDF5 and not a directory: {emb_path}"
            )
    else:
        raise FileNotFoundError(f"Embedding path does not exist: {emb_path}")

    ########## Foldseek code #########################

    def load_records(enabled=False, fasta_path=""):
        if not enabled:
            return {}
        assert fasta_path is not None
        return parse_dict(fasta_path)

    allow_foldseek = args.allow_foldseek
    allow_backbone3di = args.allow_backbone3di

    fold_record = load_records(allow_foldseek, args.foldseek_fasta)
    backbone_record = load_records(allow_backbone3di, args.backbone3di_fasta)

    backbone_vocab = build_backbone_vocab()

    foldseek3dicontext = Foldseek3diContext(
        # foldseek info
        allow_foldseek=allow_foldseek,
        fold_record=fold_record,
        fold_vocab=fold_vocab,
        # backbone info
        allow_backbone3di=allow_backbone3di,
        backbone_record=backbone_record,
        backbone_vocab=backbone_vocab,
    )

    ##################################################

    train_df = pd.read_csv(train_fi, sep="\t", header=None)
    train_df.columns = ["prot1", "prot2", "label"]

    if no_augment:
        train_p1 = train_df["prot1"]
        train_p2 = train_df["prot2"]
        train_y = torch.from_numpy(train_df["label"].values)
    else:
        train_p1 = pd.concat(
            (train_df["prot1"], train_df["prot2"]), axis=0
        ).reset_index(drop=True)
        train_p2 = pd.concat(
            (train_df["prot2"], train_df["prot1"]), axis=0
        ).reset_index(drop=True)
        train_y = torch.from_numpy(
            pd.concat((train_df["label"], train_df["label"])).values
        )

    train_dataset = PairedDataset(train_p1, train_p2, train_y)
    train_iterator = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True,
    )

    log(f"Loaded {len(train_p1)} training pairs", file=output)
    output.flush()

    test_df = pd.read_csv(test_fi, sep="\t", header=None)
    test_df.columns = ["prot1", "prot2", "label"]
    test_p1 = test_df["prot1"]
    test_p2 = test_df["prot2"]
    test_y = torch.from_numpy(test_df["label"].values)

    test_dataset = PairedDataset(test_p1, test_p2, test_y)
    test_iterator = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=False,
    )

    log(f"Loaded {len(test_p1)} test pairs", file=output)
    log("Loading embeddings...", file=output)
    output.flush()

    all_proteins = set(train_p1).union(train_p2).union(test_p1).union(test_p2)

    # Load embeddings
    embeddings: dict[str, torch.Tensor] = {}
    if embedding_mode == "pt_dir":
        embedding_loader = EmbeddingLoader(
            embedding_dir_name=emb_path, protein_names=all_proteins, num_workers=4
        )
        embeddings = embedding_loader.embeddings_cpu
    elif embedding_mode == "hdf5":
        with h5py.File(emb_path, "r") as h5fi:
            for prot_name in tqdm(all_proteins, desc="Loading HDF5 embeddings"):
                embeddings[prot_name] = torch.from_numpy(h5fi[prot_name][:, :])

    # Topsy-Turvy
    run_tt = args.run_tt
    glider_weight = args.glider_weight
    glider_thresh = args.glider_thresh * 100

    if run_tt:
        log("Running D-SCRIPT Topsy-Turvy:", file=output)
        log(f"\tglider_weight: {glider_weight}", file=output)
        log(f"\tglider_thresh: {glider_thresh}th percentile", file=output)
        log("Computing GLIDER matrix...", file=output)
        output.flush()

        glider_mat, glider_map = glide_compute_map(
            train_df[train_df.iloc[:, 2] == 1], thres_p=glider_thresh
        )
    else:
        glider_mat, glider_map = (None, None)

    # Create embedding model
    input_dim = args.input_dim

    projection_dim = args.projection_dim

    dropout_p = args.dropout_p
    embedding_model = FullyConnectedEmbed(input_dim, projection_dim, dropout=dropout_p)
    log("Initializing embedding model with:", file=output)
    log(f"\tprojection_dim: {projection_dim}", file=output)
    log(f"\tdropout_p: {dropout_p}", file=output)

    # Create contact model
    hidden_dim = args.hidden_dim
    kernel_width = args.kernel_width
    log("Initializing contact model with:", file=output)
    log(f"\thidden_dim: {hidden_dim}", file=output)
    log(f"\tkernel_width: {kernel_width}", file=output)

    proj_dim = projection_dim
    if allow_foldseek:
        proj_dim += len(fold_vocab)
    if allow_backbone3di:
        proj_dim += len(backbone_vocab)
    contact_model = ContactCNN(proj_dim, hidden_dim, kernel_width)

    # Create the full model
    do_w = not args.no_w
    do_pool = args.do_pool
    pool_width = args.pool_width
    do_sigmoid = not args.no_sigmoid
    log("Initializing interaction model with:", file=output)
    log(f"\tdo_poool: {do_pool}", file=output)
    log(f"\tpool_width: {pool_width}", file=output)
    log(f"\tdo_w: {do_w}", file=output)
    log(f"\tdo_sigmoid: {do_sigmoid}", file=output)
    model = ModelInteraction(
        embedding_model,
        contact_model,
        use_cuda,
        do_w=do_w,
        pool_size=pool_width,
        do_pool=do_pool,
        do_sigmoid=do_sigmoid,
    )
    model.use_cuda = use_cuda

    log(model, file=output)

    if args.checkpoint is not None:
        log(
            f"Loading model from checkpoint {args.checkpoint}",
            file=output,
        )
        state_dict = torch.load(args.checkpoint)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            log(
                "Warning: Loading model with strict=False due to mismatch in state_dict keys",
                file=output,
            )
            model.load_state_dict(state_dict, strict=False)

    if use_cuda:
        model.cuda()

    # Train the model
    lr = args.lr
    wd = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    inter_weight = args.interaction_weight
    cmap_weight = 1 - inter_weight
    digits = int(np.floor(np.log10(num_epochs))) + 1
    save_prefix = args.save_prefix

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    log(f'Using save prefix "{save_prefix}"', file=output)
    log(f"Training with Adam: lr={lr}, weight_decay={wd}", file=output)
    log(f"\tnum_epochs: {num_epochs}", file=output)
    log(f"\tbatch_size: {batch_size}", file=output)
    log(f"\tinteraction weight: {inter_weight}", file=output)
    log(f"\tcontact map weight: {cmap_weight}", file=output)
    output.flush()

    batch_report_fmt = "[{}/{}] training {:.1%}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}"
    epoch_report_fmt = "Finished Epoch {}/{}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, Precision={:.6}, Recall={:.6}, F1={:.6}, AUPR={:.6}"

    N = len(train_iterator) * batch_size
    for epoch in range(num_epochs):
        model.train()

        n = 0
        loss_accum = 0
        acc_accum = 0
        mse_accum = 0

        # Train batches
        for z0, z1, y in train_iterator:
            optim.zero_grad()
            loss, correct, mse, b = interaction_grad(
                model,
                z0,
                z1,
                y,
                embeddings,
                accuracy_weight=inter_weight,
                run_tt=run_tt,
                glider_weight=glider_weight,
                glider_map=glider_map,
                glider_mat=glider_mat,
                use_cuda=use_cuda,
                structural_context=foldseek3dicontext,
            )

            n += b
            delta = b * (loss - loss_accum)
            loss_accum += delta / n

            delta = correct - b * acc_accum
            acc_accum += delta / n

            delta = b * (mse - mse_accum)
            mse_accum += delta / n

            report = (n - b) // 100 < n // 100
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optim.step()
            model.clip()
           
            

            if report:
                tokens = [
                    epoch + 1,
                    num_epochs,
                    n / N,
                    loss_accum,
                    acc_accum,
                    mse_accum,
                ]
                log(batch_report_fmt.format(*tokens), file=output)

                if args.log_wandb:
                    run.log(
                        {
                            "train/loss": loss_accum,
                            "train/accuracy": acc_accum,
                            "train/mse": mse_accum,
                        }
                    )

                output.flush()

        model.eval()

        with torch.no_grad():
            (
                inter_loss,
                inter_correct,
                inter_mse,
                inter_pr,
                inter_re,
                inter_f1,
                inter_aupr,
            ) = interaction_eval(
                model, test_iterator, embeddings, use_cuda, foldseek3dicontext
            )
            tokens = [
                epoch + 1,
                num_epochs,
                inter_loss,
                inter_correct / (len(test_iterator) * batch_size),
                inter_mse,
                inter_pr,
                inter_re,
                inter_f1,
                inter_aupr,
            ]
            log(epoch_report_fmt.format(*tokens), file=output)

            if args.log_wandb:
                run.log(
                    {
                        "val/loss": inter_loss,
                        "val/accuracy": inter_correct
                        / (len(test_iterator) * batch_size),
                        "val/mse": inter_mse,
                        "val/precision": inter_pr,
                        "val/recall": inter_re,
                        "val/f1": inter_f1,
                        "val/aupr": inter_aupr,
                    }
                )

            output.flush()

            # Save the model
            if save_prefix is not None:
                save_path = (
                    save_prefix + "_epoch" + str(epoch + 1).zfill(digits) + ".sav"
                )
                log(f"Saving model to {save_path}", file=output)
                model.cpu()
                torch.save(model, save_path)
                if use_cuda:
                    model.cuda()

        output.flush()

    if save_prefix is not None:
        save_path = save_prefix + "_final.sav"
        state_dict_path = save_prefix + "_final_state_dict.sav"
        log(f"Saving final model to {save_path}", file=output)
        model.cpu()
        torch.save(model, save_path)
        torch.save(model.state_dict(), state_dict_path)

        if args.log_wandb:
            # Upload trained model as artifact
            artifact = wandb.Artifact(
                name="trained-model",
                type="model",
                description="D-SCRIPT trained interaction model",
            )
            artifact.add_file(state_dict_path)
            run.log_artifact(artifact)
            run.finish()

        if use_cuda:
            model.cuda()


def main(args):
    """
    Run training from arguments.

    :meta private:
    """

    output = args.outfile
    if output is None:
        output = sys.stdout
    else:
        output = open(output, "w")

    log(f"D-SCRIPT Version {__version__}", file=output, print_also=True)
    log(f"Called as: {' '.join(sys.argv)}", file=output, print_also=True)

    # Set the device
    device = args.device
    use_cuda = (device > -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=output,
            print_also=True,
        )
    else:
        log("Using CPU", file=output, print_also=True)
        device = "cpu"

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    train_model(args, output)

    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
