import sys
sys.path.append('../../')

import torch
import h5py
import dscript
import datetime
import pandas as pd
from tqdm import tqdm

import os

from dscript.fasta import parse_bytes
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

def file_predict(title, pairs_tsv, seqs_fasta, device=-1, modelPath = 'dscript-models/human_v1.sav', threshhold=0.5):
    """
    Given a .tsv file of candidate pairs and a .fasta file of protein sequences,
    Creates a .tsv file of interaction predictions and returns the url
    'web-server/backend/media'
    """

    # Set Outpath
    print(os.getcwd())
    outPath = f'media/predictions/{title}'

    # Set Device
    print('# Setting Device...')
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        print(f'# Using CUDA device {device} - {torch.cuda.get_device_name(device)}')
    else:
        print('# Using CPU')

    # Load Model
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

    # Load Pairs
    print('# Loading Pairs...')
    try:
        pairs = pd.read_csv(pairs_tsv, sep='\t', header=None)
        all_prots = set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1]))
    except FileNotFoundError:
        print(f'# Pairs File not found')
        return

    # Load Sequences
    try:
        print(seqs_fasta.name)
        print(seqs_fasta.size)
        # for line in seqs_fasta.chunks():
        #     print(line)
        # print(type(seqs_fasta))
        # for line in seqs_fasta:
        #     print(line)
        # file = seqs_fasta.read()
        # print(type(file))
        names, seqs = parse_bytes(seqs_fasta)
        seqDict = {n: s for n, s in zip(names, seqs)}
        print(seqDict)
    except FileNotFoundError:
        print(f'# Sequence File not found')
        return
    print('# Generating Embeddings...')
    embeddings = {}
    for n in tqdm(all_prots):
        embeddings[n] = lm_embed(seqDict[n], use_cuda)

    # Make Predictions
    print('# Making Predictions...')
    n = 0
    outPathAll = f'{outPath}.tsv'
    outPathPos = f'{outPath}.positive.tsv'
    cmap_file = h5py.File(f'{outPath}.cmaps.h5', 'w')
    model.eval()
    with open(outPathAll, 'w+') as f:
        with open(outPathPos, 'w+') as pos_f:
            with torch.no_grad():
                for _, (n0, n1) in tqdm(pairs.iloc[:, :2].iterrows(), total=len(pairs)):
                    n0 = str(n0)
                    n1 = str(n1)
                    if n % 50 == 0:
                        f.flush()
                    n += 1
                    p0 = embeddings[n0]
                    p1 = embeddings[n1]
                    if use_cuda:
                        p0 = p0.cuda()
                        p1 = p1.cuda()
                    try:
                        cm, p = model.map_predict(p0, p1)
                        p = p.item()
                        f.write(f'{n0}\t{n1}\t{p}\n')
                        if p >= threshhold:
                            pos_f.write(f'{n0}\t{n1}\t{p}\n')
                            cmap_file.create_dataset(f'{n0}x{n1}', data=cm.squeeze().cpu().numpy())
                    except RuntimeError as e:
                        print(f'{n0} x {n1} skipped - Out of Memory')
    cmap_file.close()

    return outPathAll