from pickle import FALSE
import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch

# SOURCE CODE: https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/
# ----------------------------
# POSITIVE CONTACT MAPS
# KEEP COUNT OF HOW MANY AMINO ACID RESIDES THERE ARE (AVOIDING H_)
def count_residues(residue):
    count1 = 0
    for residue_one in residue:
        if residue_one.has_id("CA"):
            count1 += 1
        else:
            count1 += 0
    return count1

# """Returns the C-alpha distance between two residues"""
def calc_residue_dist(residue_one, residue_two) :
    # if residue_one.has_id("CA") and residue_two.has_id("CA"):
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    # else: 
    #     diff_vector = numpy.array([0, 0, 0])
    return numpy.sqrt(numpy.sum(diff_vector * diff_vector))

# """Returns a matrix of C-alpha distances between two chains"""
def calc_dist_matrix(chain_one, chain_two):
    len_one = count_residues(chain_one)
    len_two = count_residues(chain_two)
    answer = numpy.zeros((len_one, len_two), numpy.float)
    print(answer.shape)
    for row, residue_one in enumerate(chain_one):
        # print(row, residue_one)
        for col, residue_two in enumerate(chain_two):
            # print(row, residue_two)
            if col < len_two and row < len_one:
                answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

# READ THROUGH PDB FILES AND PARSE THEM INTO PAIRWISE AND BINARY CONTACT MAPS
files = os.listdir("dscript/pdbs")
fastas = os.listdir("dscript/fastas")
fastas.remove(".DS_Store")
files.remove(".DS_Store")

for i in range(0, len(files)):
    files[i] = files[i][:4]
for i in range(0, len(fastas)):
    fastas[i] = fastas[i][:4]
    
# print(files)
# print(fastas)

for protein in files:
    print(protein)
    # if protein not in fastas:
    pdb_code = protein
    pdb_filename = f"dscript/pdbs/{protein}.pdb"
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    model = structure[0]

    # GET PROTEIN CHAINS
    chain = []
    for chains in structure.get_chains():
        chain.append(str(chains.get_id()))
    # print(chain)
    # CREATE H5PY FILES TO WRITE MATRICES TO
    hf_pair = h5py.File(f'dscript/pairwisecmaps/{pdb_code}:{chain[0]}x{pdb_code}:{chain[1]}', 'w')
    hf_bin = h5py.File(f'dscript/bincmaps/{pdb_code}:{chain[0]}x{pdb_code}:{chain[1]}', 'w')
    # WRITE TO FILES
    dist_matrix = calc_dist_matrix(model[chain[0]], model[chain[1]])
    contact_map = dist_matrix < 12.0
    hf_pair.create_dataset(f'{pdb_code}:{chain[0]}x{pdb_code}:{chain[1]}', data=dist_matrix)
    hf_bin.create_dataset(f'{pdb_code}:{chain[0]}x{pdb_code}:{chain[1]}', data=contact_map)

# ----------------------------
# NEGATIVE CONTACT MAPS --- UNFINISHED
# test_fi = "/Users/lynntao/opt/anaconda3/D-SCRIPT/data/pairs/human_test2train.tsv"
# test_df = pd.read_csv(test_fi, sep="\t", header=None)
# test_df.columns = ["prot1", "prot2", "label"]
# test_p1 = test_df["prot1"]
# test_p2 = test_df["prot2"]
# test_y = torch.from_numpy(test_df["label"].values)
# # print(test_y)

# prot1 = test_p1[0]
# prot2 = test_p2[0]

# print([prot1, prot2])
# # print(f">{prot1}")
# # print(f">{prot2}")

# f = open('/Users/lynntao/opt/anaconda3/D-SCRIPT/data/seqs/human2.fasta', "r")
# lines=f.readlines()
# for i in range(0, len(lines)-1):
#     print(lines[i])
#     if lines[i].strip() == f">{prot1}":
#         prot1_len = len(lines[i+1].strip())
#     if lines[i].strip() == f">{prot2}":
#         prot2_len = len(lines[i+1].strip())

# # BINARY CONTACT MAP
# contact_map = numpy.zeros((prot1_len, prot2_len), numpy.int)
# # DISTANCE MATRIX
# dist_matrix = numpy.inf((prot1_len, prot2_len))
# # PROBABILITY MATRIX
# prob_matrix = numpy.matrix((prot1_len, prot2_len))
# print(prob_matrix)

# print(numpy.finfo(float).eps)


# ----------------------------
# IMAGE DISPLAY CONTACT MAPS
# PAIRWISE DISTANCE
# dist_matrix = calc_dist_matrix(model["D"], model["M"])
# BINARY DISTANCE threshold: 12 angstroms
# contact_map = dist_matrix < 12.0

# print(contact_map)
# print(numpy.min(dist_matrix))
# print(numpy.max(dist_matrix))

# # CONTACT MAP EUCLIDIAN DISTANCES
# plt.matshow(numpy.transpose(dist_matrix))
# plt.show()

# # CONTACT MAP BINARY DISTANCES 
# plt.imshow(numpy.transpose(contact_map))
# plt.show()
