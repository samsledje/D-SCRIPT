import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
from models.interaction import ModelInteraction
# from language_model import embed_from_fasta

fi = h5py.File("2022-06-27-06:26.predictions.cmaps.h5","r")
ke = list(fi.keys())
# print(fi[ke[0]])
cmap = fi[ke[0]]
plt.imshow(cmap)
plt.show()

# SOURCE CODE: https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/
# pdb_code = "15c8"
# pdb_filename = "dscript/pdbs/15c8.pdb"
# pdb_code = "1xi4"
# pdb_filename = "dscript/1xi4.pdb" #not the full cage!


# def calc_residue_dist(residue_one, residue_two) :
#     """Returns the C-alpha distance between two residues"""
#     diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
#     return numpy.sqrt(numpy.sum(diff_vector * diff_vector))


# def calc_dist_matrix(chain_one, chain_two) :
#     """Returns a matrix of C-alpha distances between two chains"""
#     answer = numpy.zeros((len(chain_one), len(chain_two)), numpy.float)
#     for row, residue_one in enumerate(chain_one) :
#         for col, residue_two in enumerate(chain_two) :
#             answer[row, col] = calc_residue_dist(residue_one, residue_two)
#     return answer

    
# structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
# print(structure)

# model = structure[0]
# # print(list(model["H"]))

# # chain1 = model["H"]
# # for residue in chain1:
# #     if residue.id[0] != ' ':
# #         chain1.detach_child(residue.id)

# # chain2 = model["L"]
# # for residue in chain2:
# #     if residue.id[0] != ' ':
# #         chain2.detach_child(residue.id)
    
# dist_matrix = calc_dist_matrix(model["D"], model["M"])
# contact_map = dist_matrix < 12.0

# print(contact_map)
# print(numpy.min(dist_matrix))
# print(numpy.max(dist_matrix))

# pylab.matshow(numpy.transpose(dist_matrix))
# pylab.colorbar()
# pylab.show()

# pylab.imshow(numpy.transpose(contact_map))
# pylab.show()


# DSCRIPT CONTACT MAPPING
# embed_from_fasta("dscript/proteins/1a0n.fasta", "dscript/1a0n_embed.h5", device=0, verbose=False)
# h5fi = h5py.File("dscript/1a0n_embed.h5", "r")
# z_a = h5fi["1A0N:A UNP:P27986 P85A_HUMAN"]
# z_b = h5fi["1A0N:B UNP:P06241 FYN_HUMAN"]
# self = ModelInteraction()
# cm, ph = self.map_predict(z_a, z_b)
# print(cm)
# print(ph)