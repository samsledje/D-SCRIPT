from __future__ import print_function, division

import torch
import torch.utils.data

import numpy as np
import pandas as pd
import subprocess as sp
import sys
import gzip as gz

from datetime import datetime


def log(m, file=None, timestamped=True, print_also=False):
    curr_time = f"[{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}] "
    log_string = f"{curr_time if timestamped else ''}{m}"
    if file is None:
        print(log_string)
    else:
        print(log_string, file=file)
        if print_also:
            print(log_string)


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
