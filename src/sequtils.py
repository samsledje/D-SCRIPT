import sys
import gzip as gz
import pandas as pd
import numpy as np
import torch

from .fasta import parse

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

def load_data(positive_path, negative_path, alphabet, max_protein_len = 1000):
    
    x0_pos, x1_pos = load_pairs_table(positive_path, alphabet, max_protein_len)
    y_pos = torch.ones(len(x0_pos), dtype=torch.float32)
    
    x0_neg, x1_neg = load_pairs_table(negative_path, alphabet, max_protein_len)
    y_neg = torch.zeros(len(x0_neg), dtype=torch.float32)
    
    x0 = x0_pos + x0_neg
    x1 = x1_pos + x1_neg
    y = torch.cat((y_pos, y_neg), 0)
    
    return x0, x1, y

def train_dev_test_split(pos_seq, neg_seq, Npos, Nneg, prefix,train_pct=0.8,val_pct=0.1,test_pct=0.1):
    
    pos_train_ind = set(np.random.choice(len(pos_seq), int(Npos*train_pct), replace=False))
    neg_train_ind = set(np.random.choice(len(neg_seq), int(Nneg*train_pct), replace=False))
    pos_dev_ind = set(np.random.choice(len(pos_seq), int(Npos*val_pct), replace=False))
    neg_dev_ind = set(np.random.choice(len(neg_seq), int(Nneg*val_pct), replace=False))
    pos_test_ind = set(np.random.choice(len(pos_seq), int(Npos*test_pct), replace=False))
    neg_test_ind = set(np.random.choice(len(neg_seq), int(Nneg*test_pct), replace=False))
    
    pos_dev_ind = pos_dev_ind - pos_train_ind - pos_test_ind
    pos_test_ind = pos_test_ind - pos_dev_ind - pos_train_ind
    neg_dev_ind = neg_dev_ind - neg_train_ind - neg_test_ind
    neg_test_ind = neg_test_ind - neg_dev_ind - neg_train_ind
    
    print("#Positive Train:{}\n#Positive Dev:{}\n#Positive Test:{}".format(len(pos_train_ind), len(pos_dev_ind), len(pos_test_ind)))
    print("#Negative Train:{}\n#Negative Dev:{}\n#Negative Test:{}".format(len(neg_train_ind), len(neg_dev_ind), len(neg_test_ind)))
    
    pos_train_df = pos_seq.iloc[list(pos_train_ind),:]
    pos_dev_df = pos_seq.iloc[list(pos_dev_ind),:]
    pos_test_df = pos_seq.iloc[list(pos_test_ind),:]
    neg_train_df = neg_seq.iloc[list(neg_train_ind),:]
    neg_dev_df = neg_seq.iloc[list(neg_dev_ind),:]
    neg_test_df = neg_seq.iloc[list(neg_test_ind),:]
    
    suffixes = [".pos.train.txt", ".pos.dev.txt", ".pos.test.txt", ".neg.train.txt", ".neg.dev.txt", ".neg.test.txt"]
    paths = [prefix+i for i in suffixes]

    print("Writing "+paths[0])
    pos_train_df.to_csv(paths[0], sep='\t', index=False)
    print("Writing "+paths[1])
    pos_dev_df.to_csv(paths[1], sep='\t', index=False)
    print("Writing "+paths[2])
    pos_test_df.to_csv(paths[2], sep='\t', index=False)
    print("Writing "+paths[3])
    neg_train_df.to_csv(paths[3], sep='\t', index=False)
    print("Writing "+paths[4])
    neg_dev_df.to_csv(paths[4], sep='\t', index=False)
    print("Writing "+paths[5])
    neg_test_df.to_csv(paths[5], sep='\t', index=False)
    
    print("Writing pairs with names only")
    suffixes2 = [".pos.train.names.txt", ".pos.dev.names.txt", ".pos.test.names.txt", ".neg.train.names.txt", ".neg.dev.names.txt", ".neg.test.names.txt"]
    paths2 = [prefix+i for i in suffixes2]
    
    pos_train_df = pos_train_df.drop(['seq1','seq2'],axis=1)
    pos_dev_df = pos_dev_df.drop(['seq1','seq2'],axis=1)
    pos_test_df = pos_test_df.drop(['seq1','seq2'],axis=1)
    neg_train_df = neg_train_df.drop(['seq1','seq2'],axis=1)
    neg_dev_df = neg_dev_df.drop(['seq1','seq2'],axis=1)
    neg_test_df = neg_test_df.drop(['seq1','seq2'],axis=1)
    
    print("Writing "+paths2[0])
    pos_train_df.to_csv(paths2[0], sep='\t', index=False)
    print("Writing "+paths2[1])
    pos_dev_df.to_csv(paths2[1], sep='\t', index=False)
    print("Writing "+paths2[2])
    pos_test_df.to_csv(paths2[2], sep='\t', index=False)
    print("Writing "+paths2[3])
    neg_train_df.to_csv(paths2[3], sep='\t', index=False)
    print("Writing "+paths2[4])
    neg_dev_df.to_csv(paths2[4], sep='\t', index=False)
    print("Writing "+paths2[5])
    neg_test_df.to_csv(paths2[5], sep='\t', index=False)
    
    return paths