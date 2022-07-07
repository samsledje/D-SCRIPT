import gzip
from pickle import FALSE
import Bio.PDB
import numpy
import matplotlib.pyplot as plt
import h5py
import random
import os
import pandas as pd
import torch

# ip = sample.gzip

# op = open("output_file","w") 

# with gzip.open(ip,"rb") as ip_byte:
#     op.write(ip_byte.read().decode("utf-8")
#     wf.close()