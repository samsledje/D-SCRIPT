"""
    creating a training cmap tsv file with all the contact map protein pairs, all interaction = 1
"""
import Bio.PDB
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch
import csv

# files = os.listdir("dscript/bincmaps")
# files.remove(".DS_Store")
# print(files)
fi = h5py.File("data/bincmaps_train.h5","r")
ke = list(fi.keys())

with open('data/pairs/cmap_train.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for item in ke:
        # print(item)
        prot1 = item[:item.index(":")+2]
        prot2 = item[item.index(":")+3:]
        # print((prot1, prot2))
        tsv_writer.writerow([prot1, prot2, '1'])
    