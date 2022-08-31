"""
    # UNFINISHED
    # convert batches of downloaded pdb gzip files into parsable pdb protein files.
"""
import gzip
import matplotlib.pyplot as plt
import os
import pandas as pd

path = "dscript/pdbsNEW/batch-download-structures-1661040401355"
zips = os.listdir(path)
if ".DS_Store" in zips:
    zips.remove(".DS_Store")

for item in zips:
    name = item[3:7]
    op = open(f"dscript/pdbsNEW/{name}.pdb", "w")

    with gzip.open(f"{path}/{item}", "rb") as fi:
        op.write(fi.read().decode("utf-8"))
