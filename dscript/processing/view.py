from pickle import FALSE
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# fi = h5py.File("data/output_l.h5", "r")
# ke = list(fi.keys())
# print(len(ke))
# cmap = fi[ke[189]]

# n1 = np.array(cmap[:])
# print(n1)
# plt.imshow(cmap)
# plt.show()

files = os.listdir(f"dscript/pdbs_large")
print(len(files))
