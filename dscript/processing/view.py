from pickle import FALSE
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as m
import os
import seaborn as sns

# fi = h5py.File("data/test_predict.cmaps.h5", "r")
fi = h5py.File("data/test_prob.h5", "r")
ke = list(fi.keys())
print(len(ke))

for i in range(0, len(ke)):
    contact_map = np.array(fi[ke[i]])
    print(ke[i])
    print(contact_map)
    # print(contact_map)
    # print((contact_map <= 8).sum())
    # contact_map = contact_map <= 8
    plt.imshow(contact_map, cmap="Blues_r", vmin=0, vmax=1)
    plt.show()
