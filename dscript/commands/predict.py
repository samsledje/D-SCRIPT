"""
Make new predictions with a pre-trained model
"""
DO_WARN = False
TRY_PRELOAD = True

import sys, os
import torch
import h5py
import argparse
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

from dscript.alphabets import Uniprot21
from dscript.fasta import parse
from dscript.lm_embed import encode_from_fasta
from dscript.models.embedding import IdentityEmbed, SkipLSTM

THRESH = 0.7
SEP = "\t"


def add_args(parser):
    parser.add_argument(
        "-p", "--pairs", help="Candidate protein pairs to predict", required=True
    )
    parser.add_argument(
        "-f", "--fasta", help="Protein sequences in .fasta format", required=True
    )
    parser.add_argument("-m", "--model", help="Pretrained Model", required=True)
    parser.add_argument("-o", "--outfile", help="File for predictions", required=True)
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    parser.add_argument(
        "--pos_threshold",
        type=float,
        default=THRESH,
        help="Matches predicted above [pos_threshold] are written to the .positive file",
    )
    parser.add_argument("--embeddings", help="h5 file with embedded sequences")
    parser.add_argument("--sep", default=SEP, help="Separator for CSV")
    parser.add_argument(
        "--no-header", action="store_true", help="Set if CSV file has no header"
    )
    return parser


def main(args):
    csvPath = args.pairs
    fastaPath = args.fasta
    modelPath = args.model
    outPath = args.outfile
    embPath = args.embeddings
    device = args.device
    threshold = args.pos_threshold
    sep = args.sep
    no_header = args.no_header
    if no_header:
        header = None
    else:
        header = 0

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
        pairs = pd.read_csv(csvPath, sep=sep, header=header)
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
    #        embeddings = {n:torch.Tensor(embedH5[n][:,:]).unsqueeze(0) for n in all_prots}

    torch.cuda.set_device(device)
    use_cuda = device >= 0
    if device >= 0:
        print(
            "# Using CUDA device {} - {}".format(
                device, torch.cuda.get_device_name(device)
            )
        )
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
    WARN_THRESH = 1e8
    tot = len(pairs)
    if DO_WARN and (tot > WARN_THRESH):
        contin = input(
            "! Number of predictions ({}) exceeds threshold ({}). Do you want to continue? [y/n] ".format(
                int(tot), WARN_THRESH
            )
        )
        if contin.lower() != "y":
            sys.exit(0)

    alphabet = Uniprot21()

    n = 0
    with open(outPath, "w+") as f:
        with open("{}.positive".format(outPath), "w+") as g:
            with torch.no_grad():
                for _, (n0, n1) in tqdm(pairs.iterrows(), total=len(pairs)):
                    n0 = str(n0)
                    n1 = str(n1)
                    if n % 50 == 0:
                        # print('# {:.5%}'.format(n / tot))
                        f.flush()
                        g.flush()
                    n += 1
                    p0 = embeddings[n0]
                    p1 = embeddings[n1]
                    if use_cuda:
                        p0 = p0.cuda()
                        p1 = p1.cuda()

                    pred = model.predict(p0, p1)
                    p = pred.item()
                    del p0, p1, pred
                    f.write("{}\t{}\t{}\n".format(n0, n1, p))
                    if p > threshold:
                        g.write("{}\t{}\t{}\n".format(n0, n1, p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
