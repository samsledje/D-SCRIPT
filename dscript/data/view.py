"""
    visuaize contact maps
    view embed and other h5pys
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

# fi = h5py.File("dscript/pairwisecmaps/1a6u:Lx1a6u:H","r")
fi = h5py.File("2022-06-27-06:26.predictions.cmaps.h5","r")
ke = list(fi.keys())
# print(ke)
# print(ke.index("1a0n:Ax1a0n:B"))
# print(fi[ke[0]])
cmap = fi[ke[0]]
plt.imshow(cmap)
plt.show()


# fi = h5py.File("data/cmap_embed","r")
# # fi = h5py.File("2022-06-27-06:26.predictions.cmaps.h5","r")
# ke = list(fi.keys())
# print(ke)
# # for i in range(0, len(ke)):
# #     ke[i] = ke[i][:6]
# # print(ke)
# print(fi['1a7q:H'])
