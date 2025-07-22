import sys

import numpy as np
import torch
import torch.utils.data
from loguru import logger

from .loading import LoadingPool


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
    if file is not None and hasattr(file, "flush"):
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


# If keys is a dict (of key -> index) will produce a list of indices instead of a dict
# Now replaced by loading.LoadingPool; this is a wrapper for existing behavior
def load_hdf5_parallel(file_path, keys, n_jobs=-1, return_dict=True):
    """
    Load keys from hdf5 file into memory

    :param file_path: Path to hdf5 file
    :type file_path: str
    :param keys: List of keys to get
    :type keys: iterable[str]
    :return: if return_dict, a mapping of keys (proteins names) to pointers to empbeddings.
             otherwise, a list of pointers in the same order as keys
    :rtype: list
    """

    pool = LoadingPool(file_path, n_jobs)
    result = pool.load_once(keys)
    if return_dict:
        return dict(zip(keys, result))
    return result


# Parse device argument
def parse_device(device_arg, logFile):
    if device_arg.lower() == "cpu":
        device = "cpu"
        use_cuda = False
    elif device_arg.lower() == "all":
        device = -1  # Use all GPUs
        use_cuda = True
    elif device_arg.isdigit():  # Allow only nonnegative integers
        device = int(device_arg)
        use_cuda = True
    else:
        log(
            f"Invalid device argument: {device_arg}. Use 'cpu', 'all', or a GPU index.",
            file=logFile,
            print_also=True,
        )
        logFile.close()
        sys.exit(1)
    # Validate CUDA availability and device index if GPU requested
    if use_cuda:
        if not torch.cuda.is_available():
            log(
                "CUDA not available but GPU requested. Use --device cpu for CPU execution.",
                file=logFile,
                print_also=True,
            )
            logFile.close()
            sys.exit(1)
        if device >= 0 and device >= torch.cuda.device_count():
            log(
                f"Invalid device argument: {device_arg} exceeds the number of GPUs available, which is {torch.cuda.device_count()}. Please specify a valid GPU, or use --device cpu for CPU execution.",
                file=logFile,
                print_also=True,
            )
    return device


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
