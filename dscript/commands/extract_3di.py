from __future__ import annotations

import os
import shlex
import argparse
import tempfile
import subprocess as sp

from pathlib import Path
from typing import NamedTuple, Callable
from Bio import Seq, SeqRecord, SeqIO

from ..utils import log
from ..foldseek import get_3di_sequences


class Extract3DiArguments(NamedTuple):
    cmd: str
    pdb_directory: str
    out_file: str
    foldseek_path: str
    func: Callable[[Extract3DiArguments], None]


def add_args(parser):
    parser.add_argument(
        "pdb_directory", type=str, help="Path to directory with PDB files"
    )
    parser.add_argument(
        "out_file",
        type=str,
        help="Path for .fasta file containing 3Di strings",
    )
    parser.add_argument(
        "--foldseek_path",
        type=str,
        default="foldseek",
        help="Path to local Foldseek executable if not on $PATH",
    )

    return parser


def main(args):

    pdb_file_list = [
        Path(args.pdb_directory) / Path(p)
        for p in os.listdir(args.pdb_directory)
    ]

    seq_records = get_3di_sequences(
        pdb_file_list, foldseek_path=args.foldseek_path
    )
    SeqIO.write(seq_records.values(), args.out_file, "fasta-2line")

    log(f"3Di sequences written to {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    retcode = main(args)
