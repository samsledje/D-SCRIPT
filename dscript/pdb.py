import sys
import numpy as np
from .utils import RBF

resDict = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "PTR": "Y",
    "SER": "S",
    "THR": "T",
    "TPO": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "ASX": "B",
    "XLE": "J",
    "GLX": "Z",
    "UNK": "X",
    "XAA": "X",
}


def get_PDB_model(pdbid, path):
    from Bio.PDB import PDBParser

    structure = PDBParser().get_structure(pdbid, path)
    model = structure[0]
    return model


def get_CA_residues(chain):
    l = []
    for res in list(chain):
        for i in res.get_atoms():
            if i.get_name() == "CA":
                l.append(i)
                break
    return l


def calc_residue_dist(CA_one, CA_two):
    """Returns the distance between two alpha carbons"""
    diff_vector = CA_one.coord - CA_two.coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two, method="distance", cutoff=8, rbf_sigma=8):
    """Returns a matrix of C-alpha distances between two chains"""
    ch1_C_alpha = get_CA_residues(chain_one)
    ch2_C_alpha = get_CA_residues(chain_two)

    answer = np.zeros((len(ch1_C_alpha), len(ch2_C_alpha)), np.float)
    for row, CA_one in enumerate(ch1_C_alpha):
        for col, CA_two in enumerate(ch2_C_alpha):
            answer[row, col] = calc_residue_dist(CA_one, CA_two)
    if method == "binary":
        return answer < cutoff
    elif method == "closeness":
        return RBF(answer, sigma=rbf_sigma)
    elif method == "distance":
        return answer
    else:
        raise InputError


def parse_gw(gwFile):
    edge_list = []
    with open(gwFile, "r") as f:
        for line in f:
            lsplit = line.strip().split()
            if len(lsplit) == 4:
                edge_list.append((int(lsplit[0]), int(lsplit[1])))
    el = np.array(edge_list)
    N = np.max(el) + 1
    cm = np.zeros((N, N))
    for i in el:
        x, y = i
        cm[x, y] = 1
        cm[y, x] = 1
    return cm


# def cmap_from_pdb(pdbFile,chain='A',cutoff=8):
#     import subprocess as sp
#     CMAP_FILE_LOC = '/afs/csail/u/s/samsl/Applications/GR-Align/CMap'
#     outFile = pdbFile.split('.pdb')[0]+'.gw'
#     sp.run(f'{CMAP_FILE_LOC} -i {pdbFile} -o {outFile} -d {cutoff} -c {chain}'.split())
#     cm = parse_gw(outFile)
#     cm += np.eye(cm.shape[0])
#     return cm


def seq_from_model(model, chain):
    return seq_from_chain(model[chain])


def seq_from_chain(chain):
    joinseq = []
    for atom in get_CA_residues(chain):
        try:
            joinseq.append(resDict[atom.parent.get_resname()])
        except KeyError:
            continue
    return "".join(joinseq)
