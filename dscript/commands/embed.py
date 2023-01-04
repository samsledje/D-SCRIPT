"""
Generate new embeddings using pre-trained language model.
"""

from __future__ import annotations
import argparse
from ..language_model import embed_from_fasta

from typing import Callable, NamedTuple


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
    parser.add_argument(
        "--seqs", help="Sequences to be embedded", required=True
    )
    parser.add_argument(
        "-o", "--outfile", help="h5 file to write results", required=True
    )
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    return parser


def main(args):
    """
    Run embedding from arguments.

    :meta private:
    """
    inPath = args.seqs
    outPath = args.outfile
    device = args.device
    embed_from_fasta(inPath, outPath, device, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
