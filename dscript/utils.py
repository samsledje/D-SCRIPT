import multiprocessing as mp
import sys
from functools import partial

import h5py
import numpy as np
import torch
import torch.utils.data
from loguru import logger
from tqdm import tqdm


def setup_logger(log_file=None, also_stdout=False):
    """
    Setup loguru logger for D-SCRIPT.

    :param log_file: File handle or path to write logs to
    :type log_file: file handle, str, or None
    :param also_stdout: Whether to also log to stdout
    :type also_stdout: bool
    """
    # Remove default logger
    logger.remove()

    # Add file handler if log_file is provided
    if log_file is not None:
        logger.add(log_file)

    # Add stdout handler if requested or if no file specified
    if also_stdout or log_file is None:
        logger.add(sys.stdout)


def log(m, file=None, timestamped=True, print_also=False):
    """
    Legacy log function that wraps loguru for backward compatibility.

    :param m: Message to log
    :type m: str
    :param file: File handle to write to (if None, uses stdout)
    :type file: file handle or None
    :param timestamped: Whether to include timestamp (handled by loguru)
    :type timestamped: bool
    :param print_also: Whether to also print to stdout when writing to file
    :type print_also: bool
    """
    # Configure logger based on parameters
    setup_logger(log_file=file, also_stdout=print_also)

    # Log the message
    logger.info(m)

    # Flush the file if it's provided and has flush method
    if file is not None and hasattr(file, 'flush'):
        file.flush()


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
    return np.exp(-1 * (np.square(D) / (2 * sigma**2)))


def _hdf5_load_partial_func(k, file_path):
    """
    Helper function for load_hdf5_parallel
    """

    with h5py.File(file_path, "r") as fi:
        emb = torch.from_numpy(fi[k][:])
    return emb


def load_hdf5_parallel(file_path, keys, n_jobs=-1):
    """
    Load keys from hdf5 file into memory

    :param file_path: Path to hdf5 file
    :type file_path: str
    :param keys: List of keys to get
    :type keys: list[str]
    :return: Dictionary with keys and records in memory
    :rtype: dict
    """
    torch.multiprocessing.set_sharing_strategy("file_system")

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    with mp.Pool(processes=n_jobs) as pool:
        all_embs = list(
            tqdm(
                pool.imap(partial(_hdf5_load_partial_func, file_path=file_path), keys),
                total=len(keys),
            )
        )

    embeddings = {k: v for k, v in zip(keys, all_embs, strict=False)}
    return embeddings


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
    """
    Collate function for PyTorch data loader.
    """
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)
