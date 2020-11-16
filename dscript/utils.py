from __future__ import print_function, division

import torch
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import numpy as np
import pandas as pd
import subprocess as sp
import sys
import gzip as gz
from .fasta import parse
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.pairwise2 import format_alignment
from Bio.pairwise2 import align as Bio_align
from Bio.SubsMat import MatrixInfo as matlist


def plot_PR_curve(y, phat, saveFile=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    aupr = average_precision_score(y, phat)
    precision, recall, pr_thresh = precision_recall_curve(y, phat)

    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall (AUPR: {:.3})".format(aupr))
    if saveFile:
        plt.savefig(saveFile)
    else:
        plt.show()


def plot_ROC_curve(y, phat, saveFile=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    auroc = roc_auc_score(y, phat)

    fpr, tpr, roc_thresh = roc_curve(y, phat)
    print("AUROC:", auroc)

    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Receiver Operating Characteristic (AUROC: {:.3})".format(auroc))
    if saveFile:
        plt.savefig(saveFile)
    else:
        plt.show()


def RBF(D, sigma=None):
    """
    Convert distance matrix into similarity matrix using Radial Basis Function (RBF) Kernel

    :math:`RBF(x,x') = \\exp{\\frac{-((x - x')^{2}}{2\\sigma^{2}}}`

    :param D: Distance matrix
    :type D: np.ndarray
    :param sigma: Bandwith of RBF Kernel [default: :math:`\\sqrt{\\text{max}(D)}`]
    :type sigma: float
    :return: Similarity matrix
    :rtype: np.ndarray
    """
    sigma = sigma or np.sqrt(np.max(D))
    return np.exp(-1 * (np.square(D) / (2 * sigma ** 2)))


def gpu_mem(device):
    """
    Get current memory usage for GPU
    
    :param device: GPU device number
    :type device: int
    :return: memory used, memory total
    :rtype: int, int
    """
    result = sp.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
            "--id={}".format(device),
        ],
        encoding="utf-8",
    )
    gpu_memory = [int(x) for x in result.strip().split(",")]
    return gpu_memory[0], gpu_memory[1]


def align(seq1, seq2, how="local", matrix='blosum62'):
    if matrix == 'blosum62':
        matrix = matlist.blosum62
    pa = PairwiseAligner()
    pa.mode = "global"
    if how == "local":
        alignments = Bio_align.localdx(seq1, seq2, matlist.blosum62)
    elif how == "global":
        alignments = Bio_align.globaldx(seq1, seq2, matlist.blosum62)
    else:
        raise InputError("'how' must be one of ['local', 'global']")
    return alignments


def compute_sequence_similarity(seq1, seq2, how="global"):
    pa = PairwiseAligner()
    # pa.substitution_matrix = substitution_matrices.load("BLOSUM62")
    pa.mode = how
    scores = []
    raw_score = pa.score(seq1, seq2)
    norm_score = raw_score / ((len(seq1) + len(seq2)) / 2)
    return norm_score


def pack_sequences(X, order=None):

    # X = [x.squeeze(0) for x in X]

    n = len(X)
    lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]
    m = max(len(x) for x in X)

    X_block = X[0].new(n, m).zero_()

    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i, : len(x)] = x

    # X_block = torch.from_numpy(X_block)

    lengths = lengths[order]
    X = pack_padded_sequence(X_block, lengths, batch_first=True)

    return X, order


def unpack_sequences(X, order):
    X, lengths = pad_packed_sequence(X, batch_first=True)
    X_block = [None] * len(order)
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i, : lengths[i]]
    return X_block


def collate_lists(args):
    x = [a[0] for a in args]
    y = [a[1] for a in args]
    return x, y


class AllPairsDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment=None):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X) ** 2

    def __getitem__(self, k):
        n = len(self.X)
        i = k // n
        j = k % n

        x0 = self.X[i]
        x1 = self.X[j]
        if self.augment is not None:
            x0 = self.augment(x0)
            x1 = self.augment(x1)

        y = self.Y[i, j]
        # y = torch.cumprod((self.Y[i] == self.Y[j]).long(), 0).sum()

        return x0, x1, y


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, X0, X1, Y):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y
        assert len(X0) == len(X1), (
            "X0: " + str(len(X0)) + " X1: " + str(len(X1)) + " Y: " + str(len(Y))
        )
        assert len(X0) == len(Y), (
            "X0: " + str(len(X0)) + " X1: " + str(len(X1)) + " Y: " + str(len(Y))
        )

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.Y[i]


def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)


class MultinomialResample:
    def __init__(self, trans, p):
        self.p = (1 - p) * torch.eye(trans.size(0)).to(trans.device) + p * trans

    def __call__(self, x):
        # print(x.size(), x.dtype)
        p = self.p[x]  # get distribution for each x
        return torch.multinomial(p, 1).view(-1)  # sample from distribution
