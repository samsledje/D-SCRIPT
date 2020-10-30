from __future__ import print_function,division

import torch
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import numpy as np
import pandas as pd
import subprocess as sp
import sys
import gzip as gz
from .fasta import parse
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.pairwise2 import format_alignment
from Bio.pairwise2 import align as Bio_align
from Bio.SubsMat import MatrixInfo as matlist

def plot_PR_curve(y, phat, saveFile=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    aupr = average_precision_score(y, phat)
    precision, recall, pr_thresh = precision_recall_curve(y, phat)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall (AUPR: {:.3})'.format(aupr))
    if saveFile:
        plt.savefig(saveFile)
    else:
        plt.show()

def plot_ROC_curve(y, phat, saveFile=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    auroc = roc_auc_score(y, phat)

    fpr, tpr, roc_thresh = roc_curve(y, phat)
    print("AUROC:",auroc)

    plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
    plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Receiver Operating Characteristic (AUROC: {:.3})'.format(auroc))
    if saveFile:
        plt.savefig(saveFile)
    else:
        plt.show()
    
def RBF(D, sigma=None):
    """
    Convert distance matrix D into similarity matrix S using Radial Basis Function (RBF) Kernel
    RBF(x,x') = exp( -((x - x')**2 / 2sigma**@))
    """
    sigma = sigma or np.sqrt(np.max(D))
    return np.exp(-1 * (np.square(D) / (2 * sigma**2))) 

def gpu_mem(device):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = sp.check_output(
       [
           'nvidia-smi', '--query-gpu=memory.used,memory.total',
          '--format=csv,nounits,noheader', '--id={}'.format(device)
       ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split(',')]
    return gpu_memory[0], gpu_memory[1]

def align(seq1, seq2,how='local',matrix=matlist.blosum62):
    pa = PairwiseAligner()
    pa.mode = 'global'
    if how == 'local':
        alignments = Bio_align.localdx(seq1, seq2, matlist.blosum62)
    elif how == 'global':
        alignments = Bio_align.globaldx(seq1, seq2, matlist.blosum62)
    else:
        raise InputError("'how' must be one of ['local', 'global']")
    return alignments

def compute_sequence_similarity(seq1, seq2,how='global'):
    pa = PairwiseAligner()
    #pa.substitution_matrix = substitution_matrices.load("BLOSUM62")
    pa.mode = how
    scores = []
    raw_score = pa.score(seq1,seq2)
    norm_score = raw_score / ((len(seq1) + len(seq2)) / 2)
    return norm_score

def create_positive_pairs_table(fasta,links,outfile):
    
    print("Reading in sequences...")
    seqs = parse(gz.open(fasta,'rb'))
    seqs = pd.DataFrame({'protein_name': seqs[0], 'sequence': seqs[1]})
    links = pd.read_csv(links,sep=" ",compression="gzip")

    print("Filtering by experimental threshold...")
    pos_interactions = links.loc[:,['protein1','protein2']]
    pos_interactions['protein1']= pos_interactions['protein1'].str.encode('utf-8')
    pos_interactions['protein2'] = pos_interactions['protein2'].str.encode('utf-8')

    pairs = pos_interactions.merge(seqs,left_on='protein1',right_on='protein_name')
    pairs = pairs.drop(['protein_name'], axis=1).rename(columns={"sequence":"seq1"})
    pairs = pairs.merge(seqs,left_on='protein2',right_on='protein_name')
    pairs = pairs.drop(['protein_name'], axis=1).rename(columns={"sequence":"seq2"})

    print(f"Writing pairs to {outfile}...")
    with open(outfile, "wb+") as f:
        i = 0
        f.write(b'protein1\tprotein2\tseq1\tseq2\n')
        for _, link in pairs.iterrows():
            if i % 50000 == 0:
                print(i)
            f.write(link.protein1 + b'\t')
            f.write(link.protein2 + b'\t')
            f.write(link.seq1 + b'\t')
            f.write(link.seq2 + b'\n')
            i += 1

def create_random_pairs_table(fasta,outfile,number):
    
    print("Reading in sequences...")
    seqs = parse(gz.open(fasta,'rb'))
    seqs = pd.DataFrame({'protein_name': seqs[0], 'sequence': seqs[1]})
    seqs = seqs.set_index('protein_name')
    
    print(f"Randomly selecting {number} pairs...")
    neg_p1_list = np.random.choice(seqs.index, number)
    neg_p2_list = np.random.choice(seqs.index, number)

    print(f"Writing pairs to {outfile}...")
    with open(outfile, "wb+") as f:
        i = 0
        f.write(b'protein1\tprotein2\tseq1\tseq2\n')
        for p1, p2 in zip(neg_p1_list, neg_p2_list):
            if i % 50000 == 0:
                print(i)
            f.write(p1 + b'\t')
            f.write(p2 + b'\t')
            f.write(seqs.loc[p1,:].sequence + b'\t')
            f.write(seqs.loc[p2,:].sequence + b'\n')
            i += 1
            
def load_pairs_table(path, alphabet, max_protein_len):
    
    x0 = []
    x1 = []
    
    pairs_table = pd.read_csv(path, sep='\t')
    
    for s0, s1 in zip(pairs_table['seq1'], pairs_table['seq2']):
        if len(s0) < max_protein_len and len(s1)  < max_protein_len:
            s0 = s0.encode('utf-8').upper()
            s0 = torch.from_numpy(alphabet.encode(s0)).long()
            s1 = s1.encode('utf-8').upper()
            s1 = torch.from_numpy(alphabet.encode(s1)).long()
            x0.append(s0)
            x1.append(s1)
    
    return x0, x1

# def load_data(positive_path, negative_path, alphabet, max_protein_len = 1000):
    
#     x0_pos, x1_pos = load_pairs_table(positive_path, alphabet, max_protein_len)
#     y_pos = torch.ones(len(x0_pos), dtype=torch.float32)
    
#     x0_neg, x1_neg = load_pairs_table(negative_path, alphabet, max_protein_len)
#     y_neg = torch.zeros(len(x0_neg), dtype=torch.float32)
    
#     x0 = x0_pos + x0_neg
#     x1 = x1_pos + x1_neg
#     y = torch.cat((y_pos, y_neg), 0)
    
#     return x0, x1, y

def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)

def collate_cmap_pairs(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    cm = [a[2] for a in args]
    y = [a[3] for a in args]
    return x0, x1, cm, torch.stack(y)

class ContactMapDataset(torch.utils.data.Dataset):
    def __init__(self, X0, X1, C, Y, augment = None):
        self.X0 = X0
        self.X1 = X1
        self.C = C
        self.Y = Y
        assert len(X0) == len(X1)
        assert len(X0) == len(C)
        assert len(X0) == len(Y)
        self.augment = augment

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.C[i], self.Y[i]

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, X0, X1, Y):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y
        assert len(X0) == len(X1), "X0: "+str(len(X0))+" X1: "+str(len(X1))+" Y: "+str(len(Y))
        assert len(X0) == len(Y), "X0: "+str(len(X0))+" X1: "+str(len(X1))+" Y: "+str(len(Y))

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.Y[i]