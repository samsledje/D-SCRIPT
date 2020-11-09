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

def compute_sequence_similarity(seq1, seq2, how='global'):
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

def get_gpu_memory(device, report=True):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    #result = sp.check_output(
    #    [
    #        'nvidia-smi', '--query-gpu=memory.used,memory.total',
    #       '--format=csv,nounits,noheader', '--device={}'.format(device)
    #    ], encoding='utf-8')
    # Convert lines into a dictionary
    #gpu_memory = [int(x) for x in result.strip().split(',')]
    #return gpu_memory[0], gpu_memory[1]
    return 1,1

def pack_sequences(X, order=None):

    #X = [x.squeeze(0) for x in X]

    n = len(X)
    lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]
    m = max(len(x) for x in X)

    X_block = X[0].new(n,m).zero_()

    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i,:len(x)] = x

    #X_block = torch.from_numpy(X_block)

    lengths = lengths[order]
    X = pack_padded_sequence(X_block, lengths, batch_first=True)

    return X, order


def unpack_sequences(X, order):
    X,lengths = pad_packed_sequence(X, batch_first=True)
    X_block = [None]*len(order)
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i,:lengths[i]]
    return X_block


def collate_lists(args):
    x = [a[0] for a in args]
    y = [a[1] for a in args]
    return x, y


class ContactMapDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment=None, fragment=False, mi=64, ma=500):
        self.X = X
        self.Y = Y
        self.augment = augment
        self.fragment = fragment
        self.mi = mi
        self.ma = ma
        """
        if fragment: # multiply sequence occurence by expected number of fragments
            lengths = np.array([len(x) for x in X])
            mi = np.clip(lengths, None, mi)
            ma = np.clip(lengths, None, ma)
            weights = 2*lengths/(ma + mi)
            mul = np.ceil(weights).astype(int)
            X_ = []
            Y_ = []
            for i,n in enumerate(mul):
                X_ += [X[i]]*n
                Y_ += [Y[i]]*n
            self.X = X_
            self.Y = Y_
        """

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i]
        if self.fragment and len(x) > self.mi:
            mi = self.mi
            ma = min(self.ma, len(x))
            l = np.random.randint(mi, ma+1)
            i = np.random.randint(len(x)-l+1)
            xl = x[i:i+l]
            yl = y[i:i+l,i:i+l]
            # make sure there are unmasked observations
            while torch.sum(yl >= 0) == 0:
                l = np.random.randint(mi, ma+1)
                i = np.random.randint(len(x)-l+1)
                xl = x[i:i+l]
                yl = y[i:i+l,i:i+l]
            y = yl.contiguous()
            x = xl
        if self.augment is not None:
            x = self.augment(x)
        return x, y


class AllPairsDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment=None):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)**2

    def __getitem__(self, k):
        n = len(self.X)
        i = k//n
        j = k%n

        x0 = self.X[i]
        x1 = self.X[j]
        if self.augment is not None:
            x0 = self.augment(x0)
            x1 = self.augment(x1)

        y = self.Y[i,j]
        #y = torch.cumprod((self.Y[i] == self.Y[j]).long(), 0).sum()

        return x0, x1, y


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


def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)


class MultinomialResample:
    def __init__(self, trans, p):
        self.p = (1-p)*torch.eye(trans.size(0)).to(trans.device) + p*trans

    def __call__(self, x):
        #print(x.size(), x.dtype)
        p = self.p[x] # get distribution for each x
        return torch.multinomial(p, 1).view(-1) # sample from distribution
