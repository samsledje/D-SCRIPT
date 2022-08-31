"""
    # parsing PDB files into pairwise and binary contact maps
    # REMOVE ANY CA ERROR PDBS
"""
from pickle import FALSE
import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd

# SOURCE CODE: https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/
def has_CA(residue):
    """
    Return whether or not residue contains an alpha-carbon atom.

    :param residue: pass in an amino acid
    :type version: Residue object
    :return: if amino acid residue has alpha-carbons
    :rtype: boolean
    """
    return residue.has_id("CA")


def filter_chains(chain_list):
    chains_f = [[r for r in c if r.has_id("CA")] for c in chain_list]
    return chains_f


def calc_residue_dist(residue_one, residue_two, limit):
    """
    Returns the C-alpha distance between two residues

    :param residue_one: pass in an amino acid residue object
    :type residue_one: Residue object
    :param residue_two: pass in an amino acid residue object
    :type residue_two: Residue object
    :param limit: pass in a distance threshold
    :type limit: float
    :return: C-alpha distance between two residues
    :rtype: float
    """
    diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    distance = numpy.sqrt(numpy.sum(diff_vector * diff_vector))
    if distance >= float(limit):
        return float(limit)
    return distance


def calc_dist_matrix(chain_one, chain_two, limit):
    """
    Returns a contact map matrix of C-alpha distances between two chains

    :param chain_one: pass in a chain 1
    :type chain_one: Chain object
    :param chain_two: pass in a chain 2
    :type chain_two: Chain object
    :param limit: pass in a distance threshold
    :type limit: float
    :return: a contact map matrix of C-alpha distances between two chains
    :rtype: Numpy matrix
    """
    chains = filter_chains([chain_one, chain_two])
    chain1 = chains[0]
    chain2 = chains[1]

    # chain2 = filter_chains(chain_two)
    # for residue_one in chain_one:
    #     if has_CA(residue_one):
    #         chain1.append(residue_one)
    # for residue_two in chain_two:
    #     if has_CA(residue_two):
    #         chain2.append(residue_two)

    answer = numpy.zeros((len(chain1), len(chain2)), float)
    print(answer.shape)

    for i in range(0, len(chain1)):
        for j in range(0, len(chain2)):
            # print((chain1[i].get_resname(), chain2[j].get_resname()))
            answer[i][j] = calc_residue_dist(chain1[i], chain2[j], limit)

    return answer


def main():
    files = os.listdir("dscript/pdbsNEW")
    if ".DS_Store" in files:
        files.remove(".DS_Store")

    for i in range(0, len(files)):
        files[i] = files[i][:4]

    hf_pair = h5py.File(f"data/paircmaps_train.h5", "w")
    count = 0
    for protein in files:
        count += 1
        print(protein)
        print(count)
        pdb_code = protein.upper()
        pdb_filename = f"dscript/pdbsNEW/{protein}.pdb"
        structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
        model = structure[0]

        # GET PROTEIN CHAINS
        chain = []
        for chains in structure.get_chains():
            chain.append(str(chains.get_id()))
        chain = chain[:2]

        dist_matrix = calc_dist_matrix(
            model[chain[0]], model[chain[1]], 25.000
        )

        print(f"{pdb_code}:{chain[0]}x{pdb_code}:{chain[1]}")
        hf_pair.create_dataset(
            f"{pdb_code}:{chain[0]}x{pdb_code}:{chain[1]}", data=dist_matrix
        )


if __name__ == "__main__":
    main()
