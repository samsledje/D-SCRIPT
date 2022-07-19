"""
    visuaize contact maps
    view embed and other h5pys
    checks if #pdbs = #fastas = #cmaps 
"""
from pickle import FALSE
import Bio.PDB
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch

# # compare binary classification maps
# fi = h5py.File("dscript/bincmaps","r")
# ke = list(fi.keys())
# # print(ke)
# cmap = fi[ke[4]]
# print("actual binary cmap")
# n1 = np.array(cmap[:])
# print(n1)
# plt.imshow(cmap)
# plt.show()

# fi = h5py.File("2022-07-14-15:48.predictions.cmaps.h5","r")
# ke = list(fi.keys())
# # print(ke)
# cmap1 = fi[ke[4]]

# dist_matrix = np.array(cmap1[:])
# print("probability matrix")
# print(dist_matrix)
# # binarize the probability cmap
# contact_map = dist_matrix
# for i in range(len(dist_matrix)):
#     for j in range(len(dist_matrix[0])):
#         if dist_matrix[i][j] > 0.6:
#             contact_map[i][j] = 1.00
#         else:
#             contact_map[i][j] = 0.00
                            
# print("predicted binary cmap")
# print(contact_map)
# plt.imshow(contact_map)
# plt.show()


# i = 10
# # compare distance regression contact maps
# fi = h5py.File("dscript/paircmaps","r")
# ke = list(fi.keys())
# # print(ke)
# cmap = fi[ke[i]]
# print("actual distance cmap")
# n1 = np.array(cmap[:])
# print(n1)
# plt.imshow(cmap)
# plt.show()

# fi = h5py.File("2022-07-14-16:38.predictions.cmaps.h5","r")
# ke = list(fi.keys())
# # print(ke)
# cmap1 = fi[ke[i]]

# dist_matrix = np.array(cmap1[:])
# print("predicted distance cmap")
# print(dist_matrix)

# plt.imshow(dist_matrix)
# plt.show()



# fi = h5py.File("data/paircmaps","r")
# ke = list(fi.keys())
# files = os.listdir("dscript/pdbs")
# fastas = os.listdir("dscript/fastas")
# fastas.remove(".DS_Store")
# files.remove(".DS_Store")

# fi = h5py.File("data/paircmaps","r")
# ke = list(fi.keys())
# print(fi["7WHH:Ax7WHH:E"])
# print(len("PSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYAD"))
# print


fi = h5py.File("data/paircmaps_trunc","r")
ke = list(fi.keys())
# print(ke)
files = os.listdir("dscript/pdbsNEW")
fastas = os.listdir("dscript/fastasNEW")
docking = os.listdir("dscript/structures")
fastas.remove(".DS_Store")
files.remove(".DS_Store")
if ".DS_Store" in docking:
    docking.remove(".DS_Store")

print("pdbs")
print(len(files))
print("fastas")
print(len(fastas))
print("maps")
print(len(ke))
# print("docking")
# print(len(docking))

