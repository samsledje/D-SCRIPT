"""    
    # CLEAR CMAP.FASTA BEFORE RUNNING THIS FILE
    # making fasta ids simpler to match the contact maps, putting them all together in data/seqs as 1 fasta file
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

files = os.listdir("dscript/fastasTEST")
if ".DS_Store" in files:    
    files.remove(".DS_Store")
# print(files)

for item in files:
    with open('data/seqs/cmap.fasta', 'a') as out_file:
        with open(f'dscript/fastasTEST/{item}','r') as in_file:
            # print(item)
            l = in_file.read().splitlines()
            # print(l)
            for i in range(0, len(l)):
                if i%2 == 0:
                    l[i] = l[i][:7]
            # print(l)
            for line in l:
                out_file.write(line)
                out_file.write('\n')
