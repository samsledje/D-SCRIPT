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

# # # compare binary classification maps
# # fi = h5py.File("dscript/bincmaps","r")
# # ke = list(fi.keys())
# # # print(ke)
# # cmap = fi[ke[4]]
# # print("actual binary cmap")
# # n1 = np.array(cmap[:])
# # print(n1)
# # plt.imshow(cmap)
# # plt.show()

# index = 5
# fi = h5py.File("data/bincmaps_trunc","r")
# ke = list(fi.keys())
# # # print(ke)
# # # i = 0
# # # for item in ke:
# # #     print(f"{i}      {fi[item]}")
# # #     i+=1
    
# cmap = fi[ke[index]]
# # # n1 = np.array(cmap[:])
# # # print(n1)
# plt.imshow(cmap)
# plt.show()


# fi = h5py.File("data/paircmaps_trunc","r")
# ke = list(fi.keys())
# # # print(ke)
# # # i = 0
# # # for item in ke:
# # #     print(f"{i}      {fi[item]}")
# # #     i+=1
    
# cmap = fi[ke[index]]
# print(cmap)
# # # n1 = np.array(cmap[:])
# # # print(n1)
# plt.imshow(cmap)
# plt.show()


# fi = h5py.File("data/bincmaps_trunc","r")
# ke = list(fi.keys())
# print(ke)
# cmap1 = fi[ke[0]]
# plt.imshow(cmap1)
# plt.show()



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
# print(ke)

# dist_matrix = np.array(cmap1[:])
# print("predicted distance cmap")
# print(dist_matrix)

# plt.imshow(dist_matrix)
# plt.show()



# fi = h5py.File("data/bincmaps_trunc","r")
# ke = list(fi.keys())
# cmap1 = fi[ke[0]]
# plt.imshow(cmap1)
# plt.show()



# files = os.listdir("dscript/pdbs")
# fastas = os.listdir("dscript/fastas")
# fastas.remove(".DS_Store")
# files.remove(".DS_Store")

# fi = h5py.File("data/paircmaps","r")
# ke = list(fi.keys())
# print(fi["7WHH:Ax7WHH:E"])
# print(len("PSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYAD"))
# print


fi = h5py.File("data/paircmaps_train.h5","r")
ke = list(fi.keys())
# print(len(ke))
files = os.listdir("dscript/pdbs")
fastas = os.listdir("dscript/fastas")
if ".DS_Store" in fastas:
    fastas.remove(".DS_Store")
if ".DS_Store" in files:
    files.remove(".DS_Store")
    
files1 = os.listdir("dscript/pdbsTEST")
fastas1 = os.listdir("dscript/fastasTEST")
if ".DS_Store" in fastas1:
    fastas1.remove(".DS_Store")
if ".DS_Store" in files1:
    files1.remove(".DS_Store")
    
print("pdbs")
print(len(files))
print("fastas")
print(len(fastas))
print("pdbsTEST")
print(len(files1))
print("fastasTEST")
print(len(fastas1))

print("maps")
print(len(ke))
# print("docking")
# print(len(docking))

