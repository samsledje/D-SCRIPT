from pickle import FALSE
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

fi = h5py.File("data/output_test.h5", "r")
ke = list(fi.keys())
print(len(ke))

for i in range(0, len(ke)):
    cmap = fi[ke[i]]
    plt.imshow(cmap)
    plt.show()

# n1 = np.array(cmap[:])
# print(n1)


# files = os.listdir(f"dscript/pdbs_large")
# for item in files:
#     print(item[:4])
