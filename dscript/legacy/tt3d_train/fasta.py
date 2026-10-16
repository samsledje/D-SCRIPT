import os

from biotite import InvalidFileError
from biotite.sequence.io import fasta
from loguru import logger


def parse_dict(f: str):
    """
    Parse a FASTA file and return a dictionary of sequences.

    :param f: The FASTA file to parse (file-like object).
    :type f: str
    :return: A dictionary where keys are sequence names and values are sequences.
    :rtype: dict
    """

    try:
        seq_dict = fasta.get_sequences(fasta.FastaFile.read(f))
    except InvalidFileError as e:
        logger.warning(f"FASTA file is invalid or empty: {e}")
        seq_dict = {}
    for k, v in seq_dict.items():
        seq_dict[k] = str(v)

    return seq_dict


def parse(f: str):
    """
    Parse a FASTA file and return a tuple of sequence names and sequences.

    :param f: file-like object representing the FASTA file to parse.
    :type f: file-like object

    :returns: A tuple containing:
        - list of str: Sequence names.
        - list of str: Sequences.
    :rtype: (list of str, list of str)
    """

    seq_dict = parse_dict(f)

    return list(seq_dict.keys()), list(seq_dict.values())


# Iterate through FASTA, but only keep records from specified proteins
def parse_from_list(f: str, names: list[str]):
    """
    Parse a FASTA file and return a dictionary of sequences for specified names.

    :param f: The FASTA file to parse (file-like object).
    :type f: str
    :param names: List of sequence names to extract from the FASTA file.
    :type names: list of str

    :return: A dictionary where keys are sequence names and values are sequences.
    :rtype: dict
    """

    fsDict = {}
    f_iter = fasta.FastaFile.read_iter(f)
    for rkey, rseq in f_iter:
        if rkey in names:
            fsDict[rkey] = rseq
    return fsDict


def parse_directory(directory, extension=".seq"):
    """
    Parse all files in a directory with a specific extension and return their names and sequences.

    :param directory: Directory containing the files to parse.
    :type directory: str
    :param extension: File extension to filter files (default is ".seq").
    :type extension: str

    :return: A tuple containing:
        - list of str: Sequence names.
        - list of str: Sequences.
    :rtype: (list of str, list of str)
    """

    names = []
    sequences = []

    for seqPath in os.listdir(directory):
        if seqPath.endswith(extension):
            n, s = parse(f"{directory}/{seqPath}")
            names.append(n[0].strip())
            sequences.append(s[0].strip())
    return names, sequences


def write(nam, seq, f):
    """
    Write a set of sequences to a FASTA file.

    :param nam: A list of keys (sequence names).
    :type nam: list of str
    :param seq: A list of sequences.
    :type seq: list of str
    :param f: The file to write to.
    :type f: file-like object
    """

    if len(nam) != len(seq):
        logger.error("Number of names and sequences must match")
        raise ValueError("Number of names and sequences must match")

    if not len(nam):
        logger.warning("No sequences to write, skipping FASTA write.")
        return
    fasta_file = fasta.FastaFile()
    for n, s in zip(nam, seq):
        fasta_file[n] = s
    fasta_file.write(f)
    f.flush()
