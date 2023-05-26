from __future__ import annotations

import os
import shlex
import argparse
import tempfile
import subprocess as sp

from typing import NamedTuple, Callable
from Bio import Seq, SeqRecord, SeqIO

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
    parser.add_argument(
        "--foldseek_path",
        type=str,
        default="foldseek",
        help="Path to local Foldseek executable if not on $PATH",
    )

    return parser


def main(args):

    pdb_file_list = os.listdir(args.pdb_directory)
    pdb_file_string = " ".join(
        [f"{args.pdb_directory}/{i}" for i in pdb_file_list]
    )
    pdb_dir_name = hash(pdb_file_string)

    with tempfile.TemporaryDirectory() as tmpdir:

        FSEEK_BASE_CMD = f"{args.foldseek_path} createdb {pdb_file_string} {tmpdir}/{pdb_dir_name}"
        proc = sp.Popen(
            shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE
        )
        out, err = proc.communicate()

        with open(f"{tmpdir}/{pdb_dir_name}_ss", "r") as seq_file:
            seqs = [i.strip().strip("\x00") for i in seq_file]

        with open(f"{tmpdir}/{pdb_dir_name}.lookup", "r") as name_file:
            names = [i.strip().split()[1].split(".")[0] for i in name_file]

        seq_records = [
            SeqRecord.SeqRecord(Seq.Seq(s), id=n, description=n)
            for (n, s) in zip(names, seqs)
        ]
        SeqIO.write(seq_records, args.out_file, "fasta-2line")

        log(f"3Di sequences written to {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    retcode = main(args)
