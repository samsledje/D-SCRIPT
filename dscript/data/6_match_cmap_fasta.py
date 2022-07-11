from pickle import FALSE
import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch

fi = h5py.File("dscript/bincmaps","r")
ke = list(fi.keys())
# print(ke[0][:4].lower())
# print(fi[ke[0]].shape)

fastas = os.listdir("dscript/fastas")
fastas.remove(".DS_Store")
# print(fastas)
    
deletes = []

for item in ke:
    # print(fi[item].shape)
    with open(f'dscript/fastas/{item[:4].lower()}.fasta','r') as in_file:
        l = in_file.read().splitlines()
        seq1 = len(l[1])
        seq2 = len(l[3])
        # print((seq1, seq2))
        
    if (seq1 == fi[item].shape[0] and seq2 == fi[item].shape[1]) or (seq2 == fi[item].shape[0] and seq1 == fi[item].shape[1]):
        # print(f"{item} MATCHES")
        None
    else:
        # print(f"{item} DELETE")
        deletes.append(item[:4].lower())
    # print()

print(deletes)
