from pickle import FALSE
import Bio.PDB
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch

fi = h5py.File("dscript/pairwisecmaps/1a6ux1a6u","r")
# fi = h5py.File("2022-06-27-06:26.predictions.cmaps.h5","r")
ke = list(fi.keys())
# print(ke)
# print(ke.index("1a0n:Ax1a0n:B"))
# print(fi[ke[0]])
cmap = fi[ke[0]]
plt.imshow(cmap)
plt.show()


# test_fi = "/Users/lynntao/opt/anaconda3/D-SCRIPT/data/pairs/human_test2train.tsv"
# test_df = pd.read_csv(test_fi, sep="\t", header=None)
# test_df.columns = ["prot1", "prot2", "label"]
# test_p1 = test_df["prot1"]
# test_p2 = test_df["prot2"]
# test_y = torch.from_numpy(test_df["label"].values)
# print(test_p1[0])
# print(test_p2)
# print(test_y)