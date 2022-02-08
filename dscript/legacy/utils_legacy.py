import torch
import torch.utils.data

import numpy as np
import pandas as pd
import subprocess as sp
import sys
import gzip as gz
from datetime import datetime
from .fasta import parse


def log(msg, file=sys.stderr):
    """
    Log datetime-stamped message to file

    :param msg: Message to log
    :param f: Writable file object to log message to
    """
    timestr = datetime.utcnow().isoformat(sep="-", timespec="milliseconds")
    file.write(f"[{timestr}] {msg}\n")
    file.flush()


def plot_PR_curve(y, phat, saveFile=None):
    """
    Plot precision-recall curve.

    :param y: Labels
    :type y: np.ndarray
    :param phat: Predicted probabilities
    :type phat: np.ndarray
    :param saveFile: File for plot of curve to be saved to
    :type saveFile: str
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    aupr = average_precision_score(y, phat)
    precision, recall, _ = precision_recall_curve(y, phat)

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
    """
    Plot receiver operating characteristic curve.

    :param y: Labels
    :type y: np.ndarray
    :param phat: Predicted probabilities
    :type phat: np.ndarray
    :param saveFile: File for plot of curve to be saved to
    :type saveFile: str
    """
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
    Convert distance matrix into similarity matrix using Radial Basis Function (RBF) Kernel.

    :math:`RBF(x,x') = \\exp{\\frac{-(x - x')^{2}}{2\\sigma^{2}}}`

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
    Get current memory usage for GPU.

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


class PairedDataset(torch.utils.data.Dataset):
    """
    Dataset to be used by the PyTorch data loader for pairs of sequences and their labels.

    :param X0: List of first item in the pair
    :param X1: List of second item in the pair
    :param Y: List of labels
    """

    def __init__(self, X0, X1, Y):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y
        assert len(X0) == len(X1), (
            "X0: "
            + str(len(X0))
            + " X1: "
            + str(len(X1))
            + " Y: "
            + str(len(Y))
        )
        assert len(X0) == len(Y), (
            "X0: "
            + str(len(X0))
            + " X1: "
            + str(len(X1))
            + " Y: "
            + str(len(Y))
        )

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.Y[i]


def collate_paired_sequences(args):
    """
    Collate function for PyTorch data loader.
    """
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)
