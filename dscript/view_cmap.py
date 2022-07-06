from pickle import FALSE
import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
import random

# fi = h5py.File("dscript/pairwisecmaps/1xi4x1xi4","r")
fi = h5py.File("dscript/bincmaps/1xi4x1xi4","r")
ke = list(fi.keys())
# print(ke)
# print(fi[ke[0]])
cmap = fi[ke[110]]
plt.imshow(cmap)
plt.show()