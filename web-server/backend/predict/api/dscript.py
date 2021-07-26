import sys
sys.path.append('../../')

import torch
import h5py
import dscript
import datetime
import pandas as pd
from io import StringIO
from tqdm import tqdm

import os

from dscript.fasta import parse_bytes, parse_input
from dscript.language_model import lm_embed

def single_pair_predict(seq1, seq2):
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

def many_pair_predict(title, pairs_tsv, seqs_fasta, device=-1, modelPath = 'dscript-models/human_v1.sav', threshhold=0.5):
    """
    Given a .tsv file of candidate pairs and a .fasta file of protein sequences,
    Creates a .tsv file of interaction predictions and returns the url
    'web-server/backend/media'
    """

    # Set Outpath
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
        names, seqs = parse_bytes(seqs_fasta)
        seqDict = {n: s for n, s in zip(names, seqs)}
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

def all_pairs(proteins):
    """
    Generator which yields each pair of distinct proteins from a list
    """
    for i in range(len(proteins)-1):
        for j in range(i+1, len(proteins)):
            yield (proteins[i], proteins[j])

def all_pair_predict(title, seqs_fasta, device=-1, modelPath = 'dscript-models/human_v1.sav', threshhold=0.5):
    """
    Given a .fasta file of protein sequences,
    Creates a .tsv file of all interaction predictions and returns the url
    'web-server/backend/media'
    """

    # Set Outpath
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
    # print('# Loading Pairs...')
    # try:
    #     pairs = pd.read_csv(pairs_tsv, sep='\t', header=None)
    #     all_prots = set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1]))
    # except FileNotFoundError:
    #     print(f'# Pairs File not found')
    #     return

    # Load Sequences
    try:
        names, seqs = parse_bytes(seqs_fasta)
        seqDict = {n: s for n, s in zip(names, seqs)}
        all_prots = list(seqDict.keys())
        print(all_prots)
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
                # for _, (n0, n1) in tqdm(pairs.iloc[:, :2].iterrows(), total=len(pairs)):
                for (n0, n1) in tqdm(all_pairs(all_prots)):
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

def predict(pairsIndex, seqsIndex, pairs, seqs, device=-1, modelPath = 'dscript-models/human_v1.sav', threshhold=0.5):
    """
    Given specified candidate pairs and protein sequences,
    Creates a .tsv file of interaction predictions and returns the url
    within 'web-server/backend/media'
    """

    # Set Outpath
    outPath = f'media/predictions/test'

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

    # Load Sequences
    print('# Loading Sequences...')
    if seqsIndex == '1':
        try:
            names, sequences = parse_bytes(seqs)
            seqDict = {n: s for n, s in zip(names, sequences)}
        except FileNotFoundError:
            print(f'# Sequence File not found')
            return
    elif seqsIndex == '2':
        try:
            names, sequences = parse_input(seqs)
            seqDict = {n: s for n, s in zip(names, sequences)}
        except:
            return

    # Load Pairs
    print('# Loading Pairs...')
    if pairsIndex == '1':
        try:
            pairs_array = pd.read_csv(pairs, sep='\t', header=None)
            all_prots = set(pairs_array.iloc[:, 0]).union(set(pairs_array.iloc[:, 1]))
        except FileNotFoundError:
            print(f'# Pairs File not found')
            return
    elif pairsIndex == '2':
        try:
            pairs_array = pd.read_csv(StringIO(pairs), sep=',', header=None)
            all_prots = set(pairs_array.iloc[:, 0]).union(set(pairs_array.iloc[:, 1]))
        except:
            return
    elif pairsIndex == '3':
        try:
            all_prots = list(seqDict.keys())
            data = []
            for i in range(len(all_prots-1)):
                for j in range(i+1, len(all_prots)):
                    data.append([all_prots[i], all_prots[j]])
            pairs_array = pd.DataFrame(data)
        except:
            return

    # Generate Embeddings
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
                for _, (n0, n1) in tqdm(pairs_array.iloc[:, :2].iterrows(), total=len(pairs_array)):
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