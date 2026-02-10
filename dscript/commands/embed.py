"""
Generate new embeddings using pre-trained language model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import NamedTuple

from ..language_model import embed_from_fasta
from ..utils import log


class EmbeddingArguments(NamedTuple):
    cmd: str
    device: int
    outfile: str
    seqs: str
    func: Callable[[EmbeddingArguments], None]


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """
    parser.add_argument("--seqs", help="Sequences to be embedded", required=True)
    parser.add_argument(
        "-o", "--outfile", help="h5 file to write results", required=True
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Compute device to use. Options: 'cpu' or GPU index (0, 1, 2, etc.).",
    )
    return parser


def main(args):
    """
    Run embedding from arguments.

    :meta private:
    """
    inPath = args.seqs
    outPath = args.outfile
    device_arg = args.device
    if device_arg.lower() == "cpu":
        device = -1  # Refers to CPU in embed_from_fasta
    elif device_arg.isdigit():  # Allow only nonnegative integers
        device = int(device_arg)
    else:
        log(
            f"Invalid device argument: {device_arg}. Use 'cpu' or a GPU index. Using CPU."
        )
        device = -1
    embed_from_fasta(inPath, outPath, device, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
