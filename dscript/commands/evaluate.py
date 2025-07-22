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
from Bio import SeqIO
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from dscript.loading import LoadingPool

from ..utils import log

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
        "--embeddings", help="h5 file with embedded sequences", required=True
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


def get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab):
    """
    fold_record is just a dictionary {ensembl_gene_name => foldseek_sequence}
    """
    if n0 in fold_record:
        fold_seq = fold_record[n0]
        assert size_n0 == len(fold_seq)
        foldseek_enc = torch.zeros(size_n0, len(fold_vocab), dtype=torch.float32)
        for i, a in enumerate(fold_seq):
            assert a in fold_vocab
            foldseek_enc[i, fold_vocab[a]] = 1
        return foldseek_enc
    else:
        return torch.zeros(size_n0, len(fold_vocab), dtype=torch.float32)


def main(args):
    """
    Run model evaluation from arguments.

    :meta private:
    """
    ########## Foldseek code #########################3
    allow_foldseek = args.allow_foldseek
    fold_fasta_file = args.foldseek_fasta
    fold_vocab_file = args.foldseek_vocab
    add_first = not args.add_foldseek_after_projection
    fold_record = {}
    fold_vocab = None
    if allow_foldseek:
        assert fold_fasta_file is not None and fold_vocab_file is not None
        fold_fasta = SeqIO.parse(fold_fasta_file, "fasta")
        for rec in fold_fasta:
            fold_record[rec.id] = rec.seq
        with open(fold_vocab_file) as fv:
            fold_vocab = json.load(fv)
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

    embPath = args.embeddings

    # Load Pairs
    test_fi = args.test
    test_df = pd.read_csv(test_fi, sep="\t", header=None)

    if args.outfile is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    else:
        outPath = args.outfile
    outFile = open(outPath + ".predictions.tsv", "w+")

    allProteins = sorted(list(set(test_df[0]).union(test_df[1])))
    loadpool = LoadingPool(embPath, n_jobs=args.load_proc)
    embeddings = loadpool.load(allProteins)

    model.eval()
    with torch.no_grad():
        phats = []
        labels = []
        for _, (n0, n1, label) in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Predicting pairs"
        ):
            try:
                i0 = allProteins.index(n0)
                i1 = allProteins.index(n1)
                if i0 < 0 or i1 < 0:
                    raise ValueError(f"Protein {n0} or {n1} not found in embeddings")
                p0 = embeddings[i0]
                p1 = embeddings[i1]

                if use_cuda:
                    p0 = p0.cuda()
                    p1 = p1.cuda()

                if allow_foldseek:
                    f_a = get_foldseek_onehot(
                        n0, p0.shape[1], fold_record, fold_vocab
                    ).unsqueeze(0)
                    f_b = get_foldseek_onehot(
                        n1, p1.shape[1], fold_record, fold_vocab
                    ).unsqueeze(0)

                    if use_cuda:
                        f_a = f_a.cuda()
                        f_b = f_b.cuda()

                    if add_first:
                        p0 = torch.concat([p0, f_a], dim=2)
                        p1 = torch.concat([p0, f_a], dim=2)

                if allow_foldseek and (not add_first):
                    _, pred = model.map_predict(p0, p1, True, f_a, f_b)
                    pred = pred.item()
                else:
                    _, pred = model.map_predict(p0, p1)
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
