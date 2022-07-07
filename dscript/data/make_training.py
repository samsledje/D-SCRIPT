import Bio.PDB
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch
import csv

files = os.listdir("dscript/bincmaps")
files.remove(".DS_Store")
print(files)

with open('/Users/lynntao/opt/anaconda3/D-SCRIPT/data/pairs/cmap_train.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in range(0, len(files)):
        prot1 = files[i][:files[i].index(":")+2]
        prot2 = files[i][files[i].index(":")+3:]
        tsv_writer.writerow([prot1, prot2, '1'])
    