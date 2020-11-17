"""
Make new predictions with a pre-trained model.
"""
import sys, os
import torch
import h5py
import argparse
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

from dscript.alphabets import Uniprot21
from dscript.fasta import parse
from dscript.models.embedding import IdentityEmbed, SkipLSTM


def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """

    parser.add_argument("--pairs", help="Candidate protein pairs to predict", required=True)
    parser.add_argument("--seqs", help="Protein sequences in .fasta format", required=True)
    parser.add_argument("--model", help="Pretrained Model", required=True)
    parser.add_argument("-o", "--outfile", help="File for predictions")
    parser.add_argument("-d", "--device", type=int, default=-1, help="Compute device to use")
    parser.add_argument("--embeddings", help="h5 file with embedded sequences")
    parser.add_argument("--sep", default="\t", help="Separator for CSV")
    return parser


def main(args):
    """
    Run new prediction from arguments.

    :meta private:
    """
    csvPath = args.pairs
    fastaPath = args.seqs
    modelPath = args.model
    outPath = args.outfile
    embPath = args.embeddings
    device = args.device
    sep = args.sep

    if embPath:
        precomputedEmbeddings = True
    else:
        precomputedEmbeddings = False

    try:
        names, seqs = parse(open(fastaPath, "rb"))
        seqDict = {n.decode("utf-8"): s for n, s in zip(names, seqs)}
    except FileNotFoundError:
        print(f"# Sequence File {fastaPath} not found")
        sys.exit(1)

    try:
        pairs = pd.read_csv(csvPath, sep=sep, header=None)
    except FileNotFoundError:
        print(f"# Pairs File {csvPath} not found")
        sys.exit(1)

    if precomputedEmbeddings:
        all_prots = set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1]))
        print("# Preloading Embeddings...")
        embedH5 = h5py.File(embPath, "r")
        embeddings = {}
        for n in tqdm(all_prots):
            embeddings[n] = torch.from_numpy(embedH5[n][:])

    torch.cuda.set_device(device)
    use_cuda = device >= 0
    if device >= 0:
        print("# Using CUDA device {} - {}".format(device, torch.cuda.get_device_name(device)))
    else:
        print("# Using CPU")

    try:
        if use_cuda:
            model = torch.load(modelPath).cuda()
        else:
            model = torch.load(modelPath).cpu()
            model.use_cuda = False
    except FileNotFoundError:
        print(f"# Model {modelPath} not found")
        sys.exit(1)

    print("# Making Predictions...")
    n = 0
    with open(outPath, "w+") as f:
        with torch.no_grad():
            for _, (n0, n1) in tqdm(pairs.iterrows(), total=len(pairs)):
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

                p = model.predict(p0, p1).item()
                f.write("{}\t{}\t{}\n".format(n0, n1, p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
