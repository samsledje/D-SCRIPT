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
            names.append(n[0].decode("utf-8").strip())
            sequences.append(s[0].decode("utf-8").strip())
    return names, sequences


def write(nam, seq, f):
    records = [
        SeqRecord(Seq(s), id=n, name=n, description="")
        for n, s in zip(nam, seq)
    ]
    SeqIO.write(records, f, "fasta")


def count_bins(array, bins):
    # Check bins make sense
    lastB = 0
    for b in bins:
        assert b > lastB
        lastB = b
    if bins[0] > min(array) and min(array) < 0:
        bins = [min(array)] + bins
    if bins[-1] < max(array):
        bins.append(max(array))

    binDict = {b: [] for b in bins}

    for i in array:
        for b in range(len(bins)):
            if i > bins[b]:
                continue
            else:
                binDict[bins[b]].append(i)
                break

    binLens = {b: len(binDict[b]) for b in bins}

    s = 0
    for b in binDict.keys():
        s += binLens[b]
    assert s == len(array)

    return binLens
