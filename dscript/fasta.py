import os

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse(f, comment="#"):
    names = []
    sequences = []

    for record in SeqIO.parse(f, "fasta"):
        names.append(record.name)
        sequences.append(str(record.seq))

    return names, sequences


def parse_directory(directory, extension=".seq"):
    names = []
    sequences = []

    for seqPath in os.listdir(directory):
        if seqPath.endswith(extension):
            n, s = parse(f"{directory}/{seqPath}", "rb")
            names.append(n[0].strip())
            sequences.append(s[0].strip())
    return names, sequences


def write(nam, seq, f):
    if len(nam) != len(seq):
        raise ValueError("Names and sequences must have the same length.")
    records = [
        SeqRecord(Seq(s), id=n, name=n, description="")
        for n, s in zip(nam, seq, strict=False)
    ]
    SeqIO.write(records, f, "fasta")
