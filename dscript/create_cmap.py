from pickle import FALSE
import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
import random

# fi = h5py.File("2022-06-27-06:26.predictions.cmaps.h5","r")
# ke = list(fi.keys())
# print(fi[ke[0]])
# cmap = fi[ke[0]]
# plt.imshow(cmap)
# plt.show()

# SOURCE CODE: https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/
# pdb_code = "1as4"
# pdb_filename = "dscript/pdbs/1as4.pdb"
# pdb_code = "1xi4"
# pdb_filename = "dscript/pdbs/1xi4.pdb" #not the full cage!

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
# files = ["1a0n", "1a0o", "1a1o", "1a2c"]    
files = ["1a0n", "1a0o", "1a1o", "1a2c", "1a4k", "1a6w", "1a22", "1agc", "1aht", "12e8", "15c8"]  
for protein in files:
    print(protein)
    pdb_code = protein
    pdb_filename = f"dscript/pdbs/{protein}.pdb"
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    model = structure[0]

    # GET PROTEIN CHAINS
    chain = []
    for chains in structure.get_chains():
        chain.append(str(chains.get_id()))
    print(chain)
    # GET ALL POSSIBLE PAIRS OF CHAINS
    chain_pairs = []
    for i in range(0, len(chain)-1):
        for j in range(i+1, len(chain)):
            if chain[i] != chain[j] and [chain[i], chain[j]] not in chain_pairs and [chain[j], chain[i]] not in chain_pairs:
                chain_pairs.append([chain[i], chain[j]])
    print(chain_pairs)

    # CREATE H5PY FILES TO WRITE MATRICES TO
    hf_pair = h5py.File(f'dscript/pairwisecmaps/{pdb_code}x{pdb_code}', 'w')
    hf_bin = h5py.File(f'dscript/bincmaps/{pdb_code}x{pdb_code}', 'w')
    # WRITE TO FILES
    for [chain1, chain2] in chain_pairs:
        print([chain1, chain2])
        dist_matrix = calc_dist_matrix(model[chain1], model[chain2])
        contact_map = dist_matrix < 12.0
        hf_pair.create_dataset(f'{pdb_code}:{chain1}x{pdb_code}:{chain2}', data=dist_matrix)
        hf_bin.create_dataset(f'{pdb_code}:{chain1}x{pdb_code}:{chain2}', data=contact_map)

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
