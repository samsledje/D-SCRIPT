import shlex
import subprocess as sp
import tempfile

import torch
from Bio import Seq, SeqRecord

from .utils import log

fold_vocab = {
    "D": 0,
    "P": 1,
    "V": 2,
    "Q": 3,
    "A": 4,
    "W": 5,
    "K": 6,
    "E": 7,
    "I": 8,
    "T": 9,
    "L": 10,
    "F": 11,
    "G": 12,
    "S": 13,
    "M": 14,
    "H": 15,
    "C": 16,
    "R": 17,
    "Y": 18,
    "N": 19,
    "X": 20,
}


def get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab):
    """
    fold_record is just a dictionary {ensembl_gene_name => foldseek_sequence}
    """
    if n0 in fold_record:
        fold_seq = fold_record[n0]
        assert size_n0 == len(fold_seq)
        foldseek_enc = torch.zeros(size_n0, len(fold_vocab), dtype=torch.float32)
        for i, a in enumerate(fold_seq):
            assert a in fold_vocab
            foldseek_enc[i, fold_vocab[a]] = 1
        return foldseek_enc
    else:
        return torch.zeros(size_n0, len(fold_vocab), dtype=torch.float32)


def get_3di_sequences(pdb_files: list[str], foldseek_path="foldseek"):
    pdb_file_string = " ".join([str(p) for p in pdb_files])
    pdb_dir_name = hash(pdb_file_string)

    with tempfile.TemporaryDirectory() as tmpdir:
        FSEEK_BASE_CMD = (
            f"{foldseek_path} createdb {pdb_file_string} {tmpdir}/{pdb_dir_name}"
        )
        log(FSEEK_BASE_CMD)
        proc = sp.Popen(shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        out, err = proc.communicate()

        with open(f"{tmpdir}/{pdb_dir_name}_ss") as seq_file:
            seqs = [i.strip().strip("\x00") for i in seq_file]

        with open(f"{tmpdir}/{pdb_dir_name}.lookup") as name_file:
            names = [i.strip().split()[1].split(".")[0] for i in name_file]

        seq_records = {
            n: SeqRecord.SeqRecord(Seq.Seq(s), id=n, description=n)
            for (n, s) in zip(names, seqs, strict=False)
        }

        return seq_records
