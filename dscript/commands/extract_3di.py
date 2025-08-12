from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

# Use biotite for FASTA writing
import biotite.sequence.io.fasta as fasta

from ..foldseek import get_3di_sequences
from ..utils import log


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

    return parser


def main(args):
    pdb_file_list = [
        Path(args.pdb_directory) / Path(p)
        for p in os.listdir(args.pdb_directory)
        if p.endswith(".pdb") or p.endswith(".cif")
    ]

    seq_records = get_3di_sequences(pdb_file_list)
    # Convert SeqRecord to biotite sequence and write FASTA
    with open(args.out_file, "w+") as f:
        ff = fasta.FastaFile()
        for rec_k, rec_v in seq_records.items():
            ff[rec_k] = rec_v
        ff.write(f)
    log(f"3Di sequences written to {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    retcode = main(args)
