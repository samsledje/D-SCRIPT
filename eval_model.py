import sys, os
import argparse
import torch
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

import src.fasta as fa
from src.alphabets import Uniprot21
from src.embed import embed_from_fasta
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

def plot_eval_predictions(pos_phat, neg_phat, plot_prediction_distributions=True, plot_curves=True,path='figure'):

    print('Plotting Curves')
    if plot_prediction_distributions:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Distribution of Predictions')
        ax1.hist(pos_phat)
        ax1.set_xlim(0,1)
        ax1.set_title("Positive")
        ax1.set_xlabel("p-hat")
        ax2.hist(neg_phat)
        ax2.set_xlim(0,1)
        ax2.set_title("Negative")
        ax2.set_xlabel("p-hat")
        plt.savefig(path + '.phat_dist.png')
        plt.close()

    if plot_curves:
        all_phat = torch.cat((pos_phat, neg_phat),0)
        all_y = [1]*len(pos_phat) + [0]*len(neg_phat)
        precision, recall, pr_thresh = precision_recall_curve(all_y, all_phat)
        aupr = average_precision_score(all_y,all_phat)
        print("AUPR:",aupr)

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall (AUPR: {:.3})'.format(aupr))
        plt.savefig(path + '.aupr.png')
        plt.close()

        fpr, tpr, roc_thresh = roc_curve(all_y, all_phat)
        auroc = roc_auc_score(all_y,all_phat)
        print("AUROC:",auroc)

        plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Receiver Operating Characteristic (AUROC: {:.3})'.format(auroc))
        plt.savefig(path + '.auroc.png')
        plt.close()

def extract_cmap_numpy(cm):
    return cm.cpu().detach().squeeze(0).squeeze(0).numpy()

def plot_cmap(cm):
    try:
        sns.heatmap(cm)
    except:
        sns.heatmap(extract_cmap_numpy(cm))
    plt.show()

#Defaults

emb_default = '/scratch1/tbepler/data/STITCH/homo.sapiens.seqs.ssa_L1_100d_skip_lstm3x1024_uniref90_iter01000000_dlm_sim_tau0.5_augment0.05_mb64_contacts_both_mb16_0.1_0.9_0.5_iter1000000.h5'

parser = argparse.ArgumentParser('Script for predicting ')

parser.add_argument('--model', help='path to trained prediction model', required=True)
parser.add_argument('--pos-pairs', help='path to list of true positive names', required=True)
parser.add_argument('--neg-pairs', help='path to list of true negative names', required=True)

parser.add_argument('--fasta', help='path to fasta file with unembedded sequences')
parser.add_argument('--embeddings', default=emb_default, help='path to h5 file with embedded sequences')
parser.add_argument('--outfile', help='output file to write results')
parser.add_argument('--device', help='device to use', default=0)

parser.add_argument('--plot-prediction-distributions',help='create positive and negative prediction histograms',action='store_true')
parser.add_argument('--plot-curves',help='create AUPRC and AUROC curves',action='store_true')

args = parser.parse_args()

device = int(args.device)

# Load Model
torch.cuda.set_device(device)
use_cuda = device >= 0
if device >= 0:
    print('# Using CUDA device {} - {}'.format(device, torch.cuda.get_device_name(device)))
else:
    print('# Using CPU')

model_path = args.model
if use_cuda:
    model = torch.load(model_path).cuda()
else:
    model = torch.load(model_path).cpu()
    model.use_cuda = False

# Embed Sequences
if args.embeddings is None:
    if args.fasta is None:
        print('Must include one of either --fasta or --embeddings')
        sys.exit(1)
    else:
        fastaPath = args.fasta
        embeddingPath = args.fasta + 'embedding.h5'
        embed_from_fasta(fastaPath, embeddingPath, device=device)
else:
    embeddingPath = args.embeddings

h5fi = h5py.File(embeddingPath, 'r')

# Load Pairs
pos_pairs = args.pos_pairs
neg_pairs = args.neg_pairs
with open(pos_pairs,'r') as p_f:
    pos_interactions = [tuple(l.strip().split()[:2]) for l in p_f]

with open(neg_pairs,'r') as n_f:
    neg_interactions = [tuple(l.strip().split()[:2]) for l in n_f]

if args.outfile is None:
    outFile = sys.stdout
    outPath = 'figures'
else:
    outPath = args.outfile
    outFile = open(outPath+'.txt','w+')

allProteins = set()
for (p0,p1), (n0, n1) in zip(pos_interactions, neg_interactions):
    allProteins.add(p0)
    allProteins.add(p1)
    allProteins.add(n0)
    allProteins.add(n1)

seqEmbDict = {}
for i in tqdm(allProteins, desc='Loading Embeddings'):
    seqEmbDict[i] = torch.from_numpy(h5fi[i][:]).float()

print("protein1\tprotein2\tinteraction\tprobability", file=outFile)

try:
    with torch.no_grad():
        pos_phat = []
        pos_cmap = []
        for i,j in tqdm(pos_interactions,total=len(pos_interactions), desc='Positive Pairs'):
            p1 = seqEmbDict[i]
            p2 = seqEmbDict[j]
            if use_cuda:
                p1 = p1.cuda()
                p2 = p2.cuda()

            cmap, pred = model.map_predict(p1,p2)
            cm = cmap.squeeze().cpu().numpy()
            p = pred.item()
            del p1, p2, cmap, pred
            torch.cuda.empty_cache()
            pos_cmap.append(cm)
            pos_phat.append(torch.Tensor([float(p)]))
            print('{}\t{}\t1\t{:.5}'.format(i,j,p),file=outFile)

        neg_phat = []
        neg_cmap = []
        for i,j in tqdm(neg_interactions,total=len(neg_interactions), desc='Negative Pairs'):
            if use_cuda:
                p1 = torch.Tensor(h5fi[i][:]).float().cuda()
                p2 = torch.Tensor(h5fi[j][:]).float().cuda()
            else:
                p1 = torch.Tensor(h5fi[i][:]).float()
                p2 = torch.Tensor(h5fi[j][:]).float()
            cmap, pred = model.map_predict(p1,p2)
            cm = cmap.squeeze().cpu().numpy()
            p = pred.item()
            del p1, p2, cmap, pred
            torch.cuda.empty_cache()
            neg_cmap.append(cm)
            neg_phat.append(torch.Tensor([float(p)]))
            print('{}\t{}\t0\t{:.5}'.format(i,j,p),file=outFile)
except RuntimeError as e:
    print(e)
    sys.exit(1)

pos_phat = torch.stack(pos_phat, 0).squeeze(1)
neg_phat = torch.stack(neg_phat, 0).squeeze(1)
plot_eval_predictions(pos_phat, neg_phat,args.plot_prediction_distributions,args.plot_curves,outPath)

sorted_pos = [(pos_phat[i],pos_cmap[i]) for i in range(len(pos_phat))]
sorted_pos.sort(key = lambda x: x[0], reverse=True)

sorted_neg = [(neg_phat[i],neg_cmap[i]) for i in range(len(neg_phat))]
sorted_neg.sort(key = lambda x: x[0], reverse=True)

for i in range(5):
    plt.imshow(sorted_pos[i][1],vmin=0,vmax=1)
    plt.title(str(sorted_pos[i][0]))
    plt.savefig(outPath + '.top_pos_{}.png'.format(i))
    plt.close
for i in range(5):
    plt.imshow(sorted_pos[-(i+1)][1],vmin=0,vmax=1)
    plt.title(str(sorted_pos[-(i+1)][0]))
    plt.savefig(outPath + '.bottom_pos_{}.png'.format(i))
    plt.close
for i in range(5):
    plt.imshow(sorted_neg[i][1],vmin=0,vmax=1)
    plt.title(str(sorted_neg[i][0]))
    plt.savefig(outPath + '.top_neg_{}.png'.format(i))
    plt.close
for i in range(5):
    plt.imshow(sorted_neg[-(i+1)][1],vmin=0,vmax=1)
    plt.title(str(sorted_neg[-(i+1)][0]))
    plt.savefig(outPath + '.bottom_neg_{}.png'.format(i))
    plt.close

outFile.close()
h5fi.close()
