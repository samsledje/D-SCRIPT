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
# import tensorflow as tf

# fi = h5py.File("dscript/bincmaps","r")
# # fi = h5py.File("2022-06-27-06:26.predictions.cmaps.h5","r")
# ke = list(fi.keys())
# # print(ke)
# # print(ke.index("1a0n:Ax1a0n:B"))
# cmap = fi[ke[10]]
# print("actual cmap")
# plt.imshow(cmap)
# plt.show()

fi = h5py.File("2022-07-14-10:10.predictions.cmaps.h5","r")
# fi = h5py.File("2022-06-27-06:26.predictions.cmaps.h5","r")
ke = list(fi.keys())
# print(ke)
# print(ke.index("1a0n:Ax1a0n:B"))
cmap = fi[ke[10]]

n1 = np.array(cmap[:])
print(n1)

print("predicted cmap")
plt.imshow(cmap)
plt.show()

# n1 = np.array(cmap[:])
# # print(type(n1))
# print(n1)

# fi = h5py.File("data/cmap_embed","r")
# # fi = h5py.File("2022-06-27-06:26.predictions.cmaps.h5","r")
# ke = list(fi.keys())
# print(ke)
# # for i in range(0, len(ke)):
# #     ke[i] = ke[i][:6]
# # print(ke)
# print(fi['1a7q:H'])

# arr = np.array([[1, 2, 3], 
#                 [4, 5, 6],
#                 [7, 8, 9]])
# # print(arr.shape)
# # arr = arr.flatten()
# # print(type(arr))
# tens = torch.from_numpy(arr)
# tens = torch.flatten(tens)
# tens = torch.flatten(tens)
# print(tens)
# # print(type(tens))

# a = torch.zeros(5, 7, dtype=torch.float)
# a[0][0] = 2
# print(a)

# pdbs = os.listdir("dscript/pdbs")
# fastas = os.listdir("dscript/fastas")
# pdbs.remove(".DS_Store")
# fastas.remove(".DS_Store")
# print(len(pdbs))
# print(len(fastas))
# fi1 = h5py.File("dscript/paircmaps","r")
# fi2 = h5py.File("dscript/bincmaps","r")
# ke1 = list(fi1.keys())
# ke2 = list(fi2.keys())
# print(len(ke1))
# print(len(ke2))