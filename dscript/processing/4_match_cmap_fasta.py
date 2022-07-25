from pickle import FALSE
import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch

fi = h5py.File("data/paircmaps_trunc","r")
ke = list(fi.keys())
# print(ke[0][:4].lower())
# print(fi[ke[0]].shape)

fastas = os.listdir("dscript/fastasNEW")
# fastas.remove(".DS_Store")
# print(len(fastas))
    
deletes = []

for item in ke:
    # print(fi[item].shape)
    if os.path.exists(f'dscript/fastasNEW/{item[:4].lower()}.fasta'):
        with open(f'dscript/fastasNEW/{item[:4].lower()}.fasta','r') as in_file:
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
# deletes = ['1fmh', '1dzb', '1dnw', '1dnu', '1fxv', '1d5l', '1d7w', '1dtd', '1e8n']
# for item in deletes:
#     if os.path.exists(f"dscript/pdbs/{item}.pdb"):
#         os.remove(f"dscript/pdbs/{item}.pdb")
# for item in deletes:
#     if os.path.exists(f"dscript/fastas/{item}.fasta"):
#         os.remove(f"dscript/fastas/{item}.fasta")