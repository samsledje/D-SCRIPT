from pickle import FALSE
import numpy as np
import h5py
import matplotlib.pyplot as plt

fi = h5py.File("data/output.h5", "r")
ke = list(fi.keys())
# print(len(ke))
cmap = fi[ke[6]]

n1 = np.array(cmap[:])
print(n1)
plt.imshow(cmap)
plt.show()
