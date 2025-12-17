"""
Evaluate a trained model.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Callable
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from pathlib import Path
from dscript.loading import LoadingPool

from dscript.models.interaction import InteractionInputs

from ..foldseek import get_foldseek_onehot, build_backbone_vocab
from ..parallel_embedding_loader import EmbeddingLoader, add_batch_dim_if_needed
from ..fasta import parse_dict
from ..utils import log
import h5py

matplotlib.use("Agg")


class EvaluateArguments(NamedTuple):
    cmd: str
    device: int
    model: str
    embedding: str
    test: str
    func: Callable[[EvaluateArguments], None]


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """

    parser.add_argument(
        "--model",
        default="samsl/topsy_turvy_human_v1",
        type=str,
        help="Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_human_v1]",
    )
    parser.add_argument("--test", help="Test Data", required=True)
    parser.add_argument(
        "--embeddings",
        help="directory containing per-protein `.pt` embeddings or HDF5 file with embeddings",
        required=True,
    )
    parser.add_argument("-o", "--outfile", help="Output file to write results")
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    parser.add_argument(
        "--load_proc",
        type=int,
        default=16,
        help="Number of processes to use when loading embeddings (-1 = # of available CPUs, default=16). Because loading is IO-bound, values larger that the # of CPUs are allowed.",
    )

    # Foldseek arguments

    ## Foldseek arguments
    parser.add_argument(
        "--allow_foldseek",
        default=False,
        action="store_true",
        help="If set to true, adds the foldseek one-hot representation",
    )
    parser.add_argument(
        "--foldseek_fasta",
        help="foldseek fasta file containing the foldseek representation",
    )
    parser.add_argument(
        "--foldseek_vocab",
        help="foldseek vocab json file mapping foldseek alphabet to json",
    )

    parser.add_argument(
        "--add_foldseek_after_projection",
        default=False,
        action="store_true",
        help="If set to true, adds the fold seek embedding after the projection layer",
    )

    ## Backbone arguments
    parser.add_argument(
        "--allow_backbone3di",
        default=False,
        action="store_true",
        help="If set to true, adds the 12 state one-hot representation",
    )
    parser.add_argument(
        "--backbone3di_fasta",
        help="FASTA file containing the 12 state representation",
    )

    return parser


def plot_eval_predictions(labels, predictions, path="figure"):
    """
    Plot histogram of positive and negative predictions, precision-recall curve, and receiver operating characteristic curve.

    :param y: Labels
    :type y: np.ndarray
    :param phat: Predicted probabilities
    :type phat: np.ndarray
    :param path: File prefix for plots to be saved to [default: figure]
    :type path: str
    """

    pos_phat = predictions[labels == 1]
    neg_phat = predictions[labels == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Distribution of Predictions")
    ax1.hist(pos_phat)
    ax1.set_xlim(0, 1)
    ax1.set_title("Positive")
    ax1.set_xlabel("p-hat")
    ax2.hist(neg_phat)
    ax2.set_xlim(0, 1)
    ax2.set_title("Negative")
    ax2.set_xlabel("p-hat")
    plt.savefig(path + ".phat_dist.png")
    plt.close()

    precision, recall, pr_thresh = precision_recall_curve(labels, predictions)
    aupr = average_precision_score(labels, predictions)
    log(f"AUPR: {aupr}")

    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"Precision-Recall (AUPR: {aupr:.3})")
    plt.savefig(path + ".aupr.png")
    plt.close()

    fpr, tpr, roc_thresh = roc_curve(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    log(f"AUROC: {auroc}")

    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"Receiver Operating Characteristic (AUROC: {auroc:.3})")
    plt.savefig(path + ".auroc.png")
    plt.close()

def main(args):
    """
    Run model evaluation from arguments.

    :meta private:
    """
    ########## Foldseek code #########################
    allow_foldseek = args.allow_foldseek
    fold_fasta_file = args.foldseek_fasta
    fold_vocab_file = args.foldseek_vocab
    fold_record = {}
    fold_vocab = None
    if allow_foldseek:
        assert fold_fasta_file is not None and fold_vocab_file is not None
        fold_fasta = parse_dict(fold_fasta_file)
        for rec_k, rec_v in fold_fasta.items():
            fold_record[rec_k] = rec_v
        with open(fold_vocab_file) as fv:
            fold_vocab = json.load(fv)
    ########## Backbone code #########################
    allow_backbone = args.allow_backbone3di
    backbone_fasta_file = args.backbone3di_fasta
    backbone_record = {}
    backbone_vocab = None
    if allow_backbone:
        assert backbone_fasta is not None
        backbone_fasta = parse_dict(backbone_fasta_file)
        for rec_k, rec_v in backbone_fasta.items():
            backbone_record[rec_k] = rec_v
        backbone_vocab = build_backbone_vocab()

    ##################################################

    # Set Device
    device = args.device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}")
    else:
        log("Using CPU")

    # Load Model
    model_path = args.model
    if use_cuda:
        model = torch.load(model_path).cuda()
        model.use_cuda = True
    else:
        model = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        ).cpu()
        model.use_cuda = False

    emb_path = Path(args.embeddings)

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

    # Load Pairs
    test_fi = args.test

    test_df = pd.read_csv(test_fi, sep="\t", header=None)

    if args.outfile is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    else:
        outPath = args.outfile
    outFile = open(outPath + ".predictions.tsv", "w+")

    allProteins = sorted(list(set(test_df[0]).union(test_df[1])))

    # Load embeddings
    embeddings: dict[str, torch.Tensor] = {}
    if embedding_mode == "pt_dir":
        embedding_loader = EmbeddingLoader(
            embedding_dir_name=emb_path, protein_names=allProteins, num_workers=4
        )
        embeddings = embedding_loader.embeddings_cpu
    elif embedding_mode == "hdf5":
        with h5py.File(emb_path, "r") as h5fi:
            for prot_name in tqdm(allProteins, desc="Loading HDF5 embeddings"):
                embeddings[prot_name] = torch.from_numpy(h5fi[prot_name][:, :])

    model.eval()
    with torch.no_grad():
        phats = []
        labels = []
        for _, (n0, n1, label) in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Predicting pairs"
        ):
            try:

                p0 = embeddings[n0]
                p1 = embeddings[n1]

                # Ensure 3D [B, L, D]
                p0 = add_batch_dim_if_needed(p0)
                p1 = add_batch_dim_if_needed(p1)

                if use_cuda:
                    p0 = p0.cuda()
                    p1 = p1.cuda()

                f_a = f_b = b_a = b_b = None

                def build_struct_embedding(n, length, record, vocab):
                    e = get_foldseek_onehot(n, length, record, vocab).unsqueeze(0)
                    if use_cuda:
                        e = e.cuda()
                    return e

                if allow_foldseek:
                    f_a = build_struct_embedding(n0, p0.shape[1], fold_record, fold_vocab)
                    f_b = build_struct_embedding(n1, p1.shape[1], fold_record, fold_vocab)

                if allow_backbone:
                    b_a = build_struct_embedding(n0, p0.shape[1], backbone_record, backbone_vocab)
                    b_b = build_struct_embedding(n1, p1.shape[1], backbone_record, backbone_vocab)

                interactionInputs = InteractionInputs(p0, p1, embed_foldseek=allow_foldseek, f0=f_a, f1=f_b, embed_backbone=allow_backbone, b0=b_a, b1=b_b)

                _, pred = model.map_predict(interactionInputs)
                pred = pred.item()

                phats.append(pred)
                labels.append(label)
                outFile.write(f"{n0}\t{n1}\t{label}\t{pred:.5}\n")
            except Exception as e:
                sys.stderr.write(f"{n0} x {n1} - {e}")

    phats = np.array(phats)
    labels = np.array(labels)

    with open(outPath + "_metrics.txt", "w+") as f:
        aupr = average_precision_score(labels, phats)
        auroc = roc_auc_score(labels, phats)

        log(f"AUPR: {aupr}", file=f)
        log(f"AUROC: {auroc}", file=f)

    plot_eval_predictions(labels, phats, outPath)

    outFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())

