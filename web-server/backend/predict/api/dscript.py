import sys
sys.path.append('../../')

import torch
import h5py
import dscript
import datetime
import pandas as pd
from tqdm import tqdm

from dscript.fasta import parse
from dscript.language_model import lm_embed

def pair_predict(seq1, seq2):
    """
    Given a pair of protein sequences in .fasta format, outputs the probability of their interaction.
    """
    use_cuda = False # Currently using CPU

    # Load Model
    modelPath = 'dscript-models/human_v1.sav'
    print('# Loading Model...')
    try:
        if use_cuda:
            model = torch.load(modelPath).cuda()
        else:
            model = torch.load(modelPath).cpu()
            model.use_cuda = False
    except FileNotFoundError:
        print(f'# Model {modelPath} not found')
        return

    # Embed Sequences
    print('# Embedding Sequences...')
    embed1 = lm_embed(seq1, use_cuda)
    embed2 = lm_embed(seq2, use_cuda)

    # Make Prediction
    print('# Making Predictions...')
    model.eval()
    cm, p = model.map_predict(embed1, embed2)
    p = p.item()
    return round(p, 5)

