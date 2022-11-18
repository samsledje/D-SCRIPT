from pickle import FALSE
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as m
import os
import seaborn as sns

fi = h5py.File("data/output.h5", "r")
ke = list(fi.keys())
print(len(ke))


# cdict = {
#   'blue'  :  ((0.0, 8.0)),
#   'green':  ((8.0, 25.0)),
#   'red' :  ((-1.0, 0.0))
# }
# cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

# cm = m.colors.from_list('my_colormap', ["blue", "red"], N=256, gamma=1.0)

for i in range(0, len(ke)):
    contact_map = fi[ke[i]]
    print(contact_map)
    plt.imshow(contact_map, cmap="Blues_r", vmin=8)
    plt.show()

# n1 = np.array(cmap[:])
# print(n1)


# files = os.listdir(f"dscript/pdbs_large")
# for item in files:
#     print(item[:4])
