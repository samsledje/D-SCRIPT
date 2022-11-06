"""
Train a new model.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import average_precision_score as average_precision
from tqdm import tqdm
from typing import Callable, NamedTuple, Optional

import sys
import argparse
import h5py
import subprocess as sp
import numpy as np
import pandas as pd
import gzip as gz

from .. import __version__
from ..alphabets import Uniprot21
from ..glider import glide_compute_map, glider_score
from ..utils import (
    PairedDataset,
    collate_paired_sequences,
    log,
    load_hdf5_parallel,
)
from ..models.embedding import FullyConnectedEmbed
from ..models.contact import ContactCNN
from ..models.interaction import ModelInteraction
import matplotlib.pyplot as plt


class TrainArguments(NamedTuple):
    cmd: str
    device: int
    train: str
    test: str
    embedding: str
    no_augment: bool
    contact_map_train: str
    contact_map_test: str
    contact_map_mode: bool
    contact_map_embedding: str
    contact_maps: str
    contact_map_lr: float
    contact_map_lambda: float
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

    data_grp = parser.add_argument_group("Binary PPI Data")
    map_grp = parser.add_argument_group("Structural Supervision")
    proj_grp = parser.add_argument_group("Projection Module")
    contact_grp = parser.add_argument_group("Contact Module")
    inter_grp = parser.add_argument_group("Interaction Module")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")

    # Data
    data_grp.add_argument(
        "--train", required=True, help="list of training pairs"
    )
    data_grp.add_argument(
        "--test", required=True, help="list of validation/testing pairs"
    )
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

    # Contact Map
    map_grp.add_argument(
        "--contact-map-train",
        required=False,
        help="include tsv files of true contact maps for supervised training",
    )
    map_grp.add_argument(
        "--contact-map-test",
        required=False,
        help="include tsv files of true contact maps for supervised training",
    )
    map_grp.add_argument(
        "--contact-map-mode",
        required=False,
        action="store_true",
        help="enter either regression mode or classification mode",
    )
    map_grp.add_argument(
        "--contact-map-embedding",
        required=False,
        help="language model embeddings for proteins in cmap training",
    )
    map_grp.add_argument(
        "--contact-maps",
        required=False,
        help="true contact maps for pdb protein pairs",
    )
    map_grp.add_argument(
        "--contact-map-threshold",
        required=False,
        help="enter a classification distance threshold for binarization of the cmap",
    )
    map_grp.add_argument(
        "--contact-map-lr",
        required=False,
        type=float,
        default=0.00003,
        help="contact map optimizer learning rate (default: 0.00003)",
    )
    map_grp.add_argument(
        "--contact-map-lambda",
        required=False,
        type=float,
        default=0.1,
        help="weight on the similarity objective (default: 0.1)",
    )

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
        help="no use of weight matrix in interaction prediction model",
    )
    inter_grp.add_argument(
        "--no-sigmoid",
        action="store_true",
        help="no use of sigmoid activation at end of interaction model",
    )
    # don't do this
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
    misc_grp.add_argument(
        "-o", "--outfile", help="output file path (default: stdout)"
    )
    misc_grp.add_argument(
        "--save-prefix", help="path prefix for saving models"
    )
    misc_grp.add_argument(
        "-d", "--device", type=int, default=-1, help="compute device to use"
    )
    misc_grp.add_argument(
        "--checkpoint", help="checkpoint model to start training from"
    )

    return parser


def predict_cmap_interaction(model, n0, n1, tensors, use_cuda):
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
    c_map_mag = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()
        cm, ph = model.map_predict(z_a, z_b)
        p_hat.append(ph)
        c_map_mag.append(torch.mean(cm))
    p_hat = torch.stack(p_hat, 0)
    c_map_mag = torch.stack(c_map_mag, 0)
    return c_map_mag, p_hat


def cmap_interaction(model, n0, n1, tensors, use_cuda):
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
    c_map = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()
        cm, ph = model.map_predict(z_a, z_b)
        p_hat.append(ph)
        c_map.append(cm)
    p_hat = torch.stack(p_hat, 0)
    return c_map, p_hat


def predict_interaction(model, n0, n1, tensors, use_cuda):
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
    _, p_hat = predict_cmap_interaction(model, n0, n1, tensors, use_cuda)
    return p_hat


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
    c_map_mag, p_hat = predict_cmap_interaction(
        model, n0, n1, tensors, use_cuda
    )

    if use_cuda:
        y = y.cuda()
    y = Variable(y)

    p_hat = p_hat.float()
    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())

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
        accuracy_loss = (glider_weight * glider_loss) + (
            (1 - glider_weight) * bce_loss
        )
    else:
        accuracy_loss = bce_loss

    representation_loss = torch.mean(c_map_mag)
    loss = (accuracy_weight * accuracy_loss) + (
        (1 - accuracy_weight) * representation_loss
    )
    b = len(p_hat)

    # Backprop Loss
    loss.backward()

    if use_cuda:
        y = y.cpu()
        p_hat = p_hat.cpu()
        if run_tt:
            g_score = g_score.cpu()

    with torch.no_grad():
        guess_cutoff = 0.5
        p_hat = p_hat.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        y = y.float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

    return loss, correct, mse, b


def interaction_grad_cmap(
    mode_classify, model, n0, n1, y, tensors, cmaps, weight, use_cuda=True
):
    """
    Compute gradient and backpropagate loss for a contact map dataset.
    """
    c_map, p_hat = cmap_interaction(model, n0, n1, tensors, use_cuda)

    if use_cuda:
        y = y.cuda()
    y = Variable(y)

    # CONTACT MAP LOSS FUNCTION
    if mode_classify:
        loss_fn = torch.nn.BCELoss()
    else:
        loss_fn = torch.nn.MSELoss()
    losses = []

    for i in range(0, len(n0)):
        true_cmap = torch.from_numpy(cmaps[f"{n0[i]}x{n1[i]}"])
        true_cmap_fl = (true_cmap).float()
        c_map_sq = torch.squeeze(c_map[i])
        c_map_fl = (c_map_sq).float()

        if use_cuda:
            true_cmap_fl = true_cmap_fl.cuda()
            c_map_fl = c_map_fl.cuda()
        # true_cmap_fl = Variable(true_cmap_fl)
        # c_map_fl = Variable(c_map_fl)

        # print(f"Square Loss: {loss_fn(c_map[i].double(), true_cmap.double())}")
        # print(f"Flat Loss: {loss_fn(c_map_fldb, true_cmap_fldb)}")
        map_loss = loss_fn(c_map_fl, true_cmap_fl)
        losses.append(map_loss)

    # prediction interaction loss
    p_hat = p_hat.float()
    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())
    cmap_loss = torch.mean(torch.stack(losses))
    loss = (weight * bce_loss) + ((1 - weight) * cmap_loss)
    b = len(p_hat)

    loss.backward()

    if use_cuda:
        y = y.cpu()
        p_hat = p_hat.cpu()

    with torch.no_grad():
        guess_cutoff = 0.5
        p_hat = p_hat.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        y = y.float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

    # return loss, correct, mse, b
    # keep mse, could monitor magnitude of cmap
    # pearson correlation between two contact maps
    # decide which metrics are good here - interaction AUPR
    return loss, mse, correct, b


def interaction_eval(model, test_iterator, tensors, use_cuda, epoch_loss):
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
    p_hat = []
    true_y = []

    for n0, n1, y in test_iterator:
        p_hat.append(predict_interaction(model, n0, n1, tensors, use_cuda))
        true_y.append(y)

    y = torch.cat(true_y, 0)
    p_hat = torch.cat(p_hat, 0)

    if use_cuda:
        y.cuda()
        p_hat = torch.Tensor([x.cuda() for x in p_hat])
        p_hat.cuda()

    loss = F.binary_cross_entropy(p_hat.float(), y.float()).item()
    epoch_loss.append(loss)
    b = len(y)

    with torch.no_grad():
        guess_cutoff = torch.Tensor([0.5]).float()
        p_hat = p_hat.float()
        y = y.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

        tp = torch.sum(y * p_hat).item()
        pr = tp / torch.sum(p_hat).item()
        re = tp / torch.sum(y).item()
        f1 = 2 * pr * re / (pr + re)

    y = y.cpu().numpy()
    p_hat = p_hat.data.cpu().numpy()

    aupr = average_precision(y, p_hat)

    return loss, correct, mse, pr, re, f1, aupr, epoch_loss


def train_model(args, output):
    # Create data sets

    batch_size = args.batch_size
    use_cuda = (args.device > -1) and torch.cuda.is_available()
    train_fi = args.train
    test_fi = args.test
    no_augment = args.no_augment

    embedding_h5 = args.embedding

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
    embeddings = load_hdf5_parallel(embedding_h5, all_proteins)

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

    # CONTACT MAP DATA LOADING
    mode_classify = args.contact_map_mode
    if args.contact_map_train is not None:
        fimaps = args.contact_maps
        cmap_train = args.contact_map_train
        cmap_test = args.contact_map_test
        cmap_embeddings = args.contact_map_embedding

        threshold = args.contact_map_threshold
        if threshold is not None:
            threshold = float(threshold)

        log("Loading training pairs for contact maps", file=output)
        output.flush()

        cmap_trainfi = pd.read_csv(cmap_train, sep="\t", header=None)
        cmap_trainfi.columns = ["prot1", "prot2", "label"]
        # create paired dataset for contact map proteins
        cmap_train_p1 = cmap_trainfi["prot1"]
        cmap_train_p2 = cmap_trainfi["prot2"]
        cmap_train_y = torch.from_numpy(cmap_trainfi["label"].values)

        cmap_train_dataset = PairedDataset(
            cmap_train_p1, cmap_train_p2, cmap_train_y
        )
        cmap_train_iterator = torch.utils.data.DataLoader(
            cmap_train_dataset,
            batch_size=batch_size,
            collate_fn=collate_paired_sequences,
            shuffle=True,
        )

        log(
            f"Loaded {len(cmap_train_p1)} contact map training pairs",
            file=output,
        )
        log("Loading testing pairs for contact maps", file=output)
        output.flush()

        cmap_testfi = pd.read_csv(cmap_test, sep="\t", header=None)
        cmap_testfi.columns = ["prot1", "prot2", "label"]
        cmap_test_p1 = cmap_testfi["prot1"]
        cmap_test_p2 = cmap_testfi["prot2"]
        cmap_test_y = torch.from_numpy(cmap_testfi["label"].values)

        cmap_test_dataset = PairedDataset(
            cmap_test_p1, cmap_test_p2, cmap_test_y
        )
        cmap_test_iterator = torch.utils.data.DataLoader(
            cmap_test_dataset,
            batch_size=batch_size,
            collate_fn=collate_paired_sequences,
            shuffle=False,
        )

        log(f"Loaded {len(cmap_test_p1)} test pairs", file=output)
        log("Loading dictionary of contact maps", file=output)
        output.flush()

        # load in dictionary of contact maps
        maps = {}
        fi = h5py.File(fimaps, "r")
        for i in range(0, len(cmap_train_p1)):
            item = f"{cmap_train_p1[i]}x{cmap_train_p2[i]}"
            c_map = np.array(fi[item][:])
            if not mode_classify:
                maps[f"{item}"] = c_map
            else:
                contact_map = (c_map <= threshold).astype(float)
                maps[f"{item}"] = contact_map

        log(f"Loaded {len(maps.keys())} contact maps", file=output)
        output.flush()

        # load in dictionary of cmap protein embeddings
        # print(cmap_embeddings)
        log("Loading embeddings of contact maps", file=output)
        output.flush()
        cmap_h5fi = h5py.File(cmap_embeddings, "r")
        cmap_embeddings = {}
        cmap_proteins = (
            set(cmap_train_p1)
            .union(cmap_train_p2)
            .union(cmap_test_p1)
            .union(cmap_test_p2)
        )
        for prot_name in tqdm(cmap_proteins):
            cmap_embeddings[prot_name] = torch.from_numpy(
                cmap_h5fi[prot_name][:, :]
            )

    if mode_classify:
        activation = nn.Sigmoid()
    else:
        activation = nn.ReLU()

    if args.checkpoint is None:

        # Create embedding model
        input_dim = args.input_dim
        projection_dim = args.projection_dim
        dropout_p = args.dropout_p
        embedding_model = FullyConnectedEmbed(
            input_dim, projection_dim, dropout=dropout_p
        )
        log("Initializing embedding model with:", file=output)
        log(f"\tprojection_dim: {projection_dim}", file=output)
        log(f"\tdropout_p: {dropout_p}", file=output)

        # Create contact model
        hidden_dim = args.hidden_dim
        kernel_width = args.kernel_width
        log("Initializing contact model with:", file=output)
        log(f"\thidden_dim: {hidden_dim}", file=output)
        log(f"\tkernel_width: {kernel_width}", file=output)

        contact_model = ContactCNN(
            projection_dim, hidden_dim, kernel_width, activation
        )

        # Create the full model
        do_w = not args.no_w
        do_pool = args.do_pool
        pool_width = args.pool_width
        do_sigmoid = not args.no_sigmoid
        log("Initializing interaction model with:", file=output)
        log(f"\tdo_pool: {do_pool}", file=output)
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

        log(model, file=output)

    else:
        log(
            "Loading model from checkpoint {}".format(args.checkpoint),
            file=output,
        )
        model = torch.load(args.checkpoint)
        model.use_cuda = use_cuda

    if use_cuda:
        model.cuda()

    # Train the model
    lr = args.lr
    wd = args.weight_decay
    map_lr = args.contact_map_lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    inter_weight = args.interaction_weight
    map_inter_weight = args.contact_map_lambda
    cmap_weight = 1 - inter_weight
    digits = int(np.floor(np.log10(num_epochs))) + 1
    save_prefix = args.save_prefix

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    # CONTACT MAP OPTIMIZER
    # make separate learning rates  lr
    optim_cmap = torch.optim.Adam(params, lr=map_lr, weight_decay=wd)

    log(f'Using save prefix "{save_prefix}"', file=output)
    log(f"Training with Adam: lr={lr}, weight_decay={wd}", file=output)
    log(
        f"Contact maps -- Training with Adam: lr={map_lr}, weight_decay={wd}",
        file=output,
    )
    log(f"\tnum_epochs: {num_epochs}", file=output)
    log(f"\tbatch_size: {batch_size}", file=output)
    log(f"\tinteraction weight: {inter_weight}", file=output)
    log(f"\tcontact map weight: {cmap_weight}", file=output)
    log(f"\tcmap training interaction weight: {map_inter_weight}", file=output)
    output.flush()

    batch_report_fmt = (
        "[{}/{}] training {:.1%}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}"
    )
    epoch_report_fmt = "Finished Epoch {}/{}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, Precision={:.6}, Recall={:.6}, F1={:.6}, AUPR={:.6}"
    epoch_report_cmap = "Finished Contact Map Epoch {}/{}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, Precision={:.6}, Recall={:.6}, F1={:.6}, AUPR={:.6}"

    loss_vals = []
    acc_vals = []

    N = len(train_iterator) * batch_size
    if args.contact_map_train is not None:
        N_cmap = len(cmap_train_iterator) * batch_size
    for epoch in range(num_epochs):
        epoch_loss = []
        loss_cmap = []

        model.train()

        n = 0
        n_cmap = 0
        loss_accum = 0
        acc_accum = 0
        mse_accum = 0

        # Train batches
        for (z0, z1, y) in train_iterator:

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
            )

            n += b
            delta = b * (loss - loss_accum)
            loss_accum += delta / n

            delta = correct - b * acc_accum
            acc_accum += delta / n

            delta = b * (mse - mse_accum)
            mse_accum += delta / n

            report = (n - b) // 100 < n // 100

            optim.step()
            optim.zero_grad()
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
                output.flush()

        # CONTACT MAP TRAINING LOOP
        if args.contact_map_train is not None:
            loss_accum_cmap = 0
            acc_accum_cmap = 0
            mse_accum_cmap = 0

            # Train batches
            for (z0, z1, y) in cmap_train_iterator:

                loss, mse, correct, b = interaction_grad_cmap(
                    mode_classify,
                    model,
                    z0,
                    z1,
                    y,
                    cmap_embeddings,
                    maps,
                    # add as param
                    weight=map_inter_weight,
                    use_cuda=use_cuda,
                )

                n_cmap += b
                delta = b * (loss - loss_accum_cmap)
                loss_accum_cmap += delta / n_cmap

                delta = correct - b * acc_accum_cmap
                acc_accum_cmap += delta / n_cmap

                delta = b * (mse - mse_accum_cmap)
                mse_accum_cmap += delta / n_cmap

                report = (n_cmap - b) // 100 < n_cmap // 100
                # print(report)

                optim_cmap.step()
                optim_cmap.zero_grad()
                model.clip()

                if report:
                    tokens = [
                        epoch + 1,
                        num_epochs,
                        n_cmap / N_cmap,
                        loss_accum_cmap,
                        acc_accum_cmap,
                        mse_accum_cmap,
                    ]
                    log(batch_report_fmt.format(*tokens), file=output)
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
                epoch_loss,
            ) = interaction_eval(
                model, test_iterator, embeddings, use_cuda, epoch_loss
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
            output.flush()

            # cmap evaluation
            if args.contact_map_train is not None:
                (
                    inter_loss,
                    inter_correct,
                    inter_mse,
                    inter_pr,
                    inter_re,
                    inter_f1,
                    inter_aupr,
                    loss_cmap,
                ) = interaction_eval(
                    model,
                    cmap_test_iterator,
                    cmap_embeddings,
                    use_cuda,
                    loss_cmap,
                )
                tokens = [
                    epoch + 1,
                    num_epochs,
                    inter_loss,
                    inter_correct / (len(cmap_test_iterator) * batch_size),
                    inter_mse,
                    inter_pr,
                    inter_re,
                    inter_f1,
                    inter_aupr,
                ]
                log(epoch_report_cmap.format(*tokens), file=output)
                output.flush()

        loss_vals.append(sum(epoch_loss) / len(epoch_loss))
        acc_vals.append(inter_correct / (len(test_iterator) * batch_size))

        with torch.no_grad():
            # Save the model
            if save_prefix is not None:
                save_path = (
                    save_prefix
                    + "_epoch"
                    + str(epoch + 1).zfill(digits)
                    + ".sav"
                )
                log(f"Saving model to {save_path}", file=output)
                model.cpu()
                torch.save(model, save_path)
                if use_cuda:
                    model.cuda()

        output.flush()

    # print(loss_vals)
    # print(acc_vals)
    # plt.plot(loss_vals, [1, 2, 3, 4, 5], 'b', label='validation loss')
    # plt.show()

    if save_prefix is not None:
        save_path = save_prefix + "_final.sav"
        log(f"Saving final model to {save_path}", file=output)
        model.cpu()
        torch.save(model, save_path)
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
    log(f'Called as: {" ".join(sys.argv)}', file=output, print_also=True)

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

    train_model(args, output)
    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
