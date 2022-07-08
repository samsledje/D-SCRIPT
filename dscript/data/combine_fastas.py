import Bio.PDB
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch
import csv

files = os.listdir("dscript/fastas")
files.remove(".DS_Store")
print(files)

for item in files:
    with open('data/seqs/cmap.fasta', 'a') as out_file:
        with open(f'dscript/fastas/{item}','r') as in_file:
            for line in in_file:
                out_file.write(line)
