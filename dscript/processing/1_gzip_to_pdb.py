"""
    # UNFINISHED
    # convert batches of downloaded pdb gzip files into parsable pdb protein files. 
"""
import gzip
# from pickle import FALSE
import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch

path = "dscript/batches4"
zips = os.listdir(path)
# print(zips)
zips.remove(".DS_Store")

# print(zips)

for item in zips:
    name = item[3:7]
    # print(name)
    # print(name)
    op = open(f"dscript/pdbsNEW/{name}.pdb","w") 

    with gzip.open(f"{path}/{item}","rb") as fi:
        op.write(fi.read().decode("utf-8"))
    