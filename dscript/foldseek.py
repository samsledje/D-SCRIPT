import biotite.structure as struc
import biotite.structure.alphabet as alphabet
import biotite.structure.io as strucio
import torch
from loguru import logger

from .utils import log

from dataclasses import dataclass


@dataclass
class Foldseek3diContext:
    allow_foldseek: bool = False
    allow_backbone3di: bool = False

    fold_record: dict = None
    fold_vocab: dict = None

    backbone_record: dict = None
    backbone_vocab: dict = None


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


def build_backbone_vocab(size=12):
    """
    Build a reduced amino-acid vocabulary mapping for backbone modeling.

    This function selects the first `size` amino acids from a fixed alphabet
    (`"ACDEFGHIKLMPQNRSTVWY"`) and returns a dictionary mapping each selected
    amino acid to its corresponding value in `fold_vocab`. The backbone
    sequence generation script uses "H" as invalid residue replacement.

    :param size: Number of amino acids to include from the start of the
                 alphabet. Must be between 1 and 20.
    :type size: int, optional
    :returns: A dictionary mapping each selected amino-acid letter in range to
              an index.
    :rtype: dict
    :raises ValueError: If `size` is less than 1 or greater than the number
                        of amino acids in the alphabet (20).
    """

    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    if size < 1 or size > len(aa_alphabet):
        raise ValueError(f"Size must be between 1 and {len(aa_alphabet)}")
    letters = aa_alphabet[:size]
    vocab = {aa: i for i, aa in enumerate(letters)}
    return vocab


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


def get_3di_sequences(pdb_files: list[str]):
    """
    Extract 3Di sequences from PDB/mmCIF files using biotite.structure.alphabet.to_3di(atoms).
    Returns a dict {basename: SeqRecord}.

    At this time, this function will only extract a 3Di sequence for the first chain in each PDB file.
    If you need to extract multiple chains, you will need to modify this function. This is to maintain
    consistent naming support with the rest of D-SCRIPT training and inference scripts, as the current
    requirement is that pdb file names match fasta header names.
    """
    seq_records = {}
    for pdb_path in pdb_files:
        basename = str(pdb_path).split("/")[-1].split(".")[0]
        try:
            atoms = strucio.load_structure(str(pdb_path))
            atoms = atoms[struc.filter_amino_acids(atoms)]
            chains = sorted(list(set(atoms.chain_id)))
            first_chain = chains[0]
            chain_atoms = atoms[atoms.chain_id == first_chain]
            if len(chain_atoms) == 0:
                logger.warning(f"No atoms found for chain {first_chain} in {pdb_path}")
            seq_3di, idx = alphabet.to_3di(chain_atoms)
            seq_records[basename] = str(seq_3di[0]).upper()
        except Exception as e:
            log(f"Error processing {pdb_path}: {e}")
            continue
    return seq_records
