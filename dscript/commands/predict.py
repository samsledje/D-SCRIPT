"""
Make new predictions with a pre-trained model. One of --seqs or --embeddings is required.
"""
import sys, os
import torch
import h5py
import argparse
import datetime
import numpy as np
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

from dscript.alphabets import Uniprot21
from dscript.fasta import parse
from dscript.language_model import lm_embed
from dscript.utils import log

def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """

    parser.add_argument("--pairs", help="Candidate protein pairs to predict", required=True)
    parser.add_argument("--model", help="Pretrained Model", required=True)
    parser.add_argument("--seqs", help="Protein sequences in .fasta format")
    parser.add_argument("--embeddings", help="h5 file with embedded sequences")
    parser.add_argument("-o", "--outfile", help="File for predictions")
    parser.add_argument("-d", "--device", type=int, default=-1, help="Compute device to use")
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.5,
        help="Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]",
    )
    return parser


def main(args):
    """
    Run new prediction from arguments.

    :meta private:
    """
    if args.seqs is None and args.embeddings is None:
        print("One of --seqs or --embeddings is required.")
        sys.exit(0)

    csvPath = args.pairs
    modelPath = args.model
    outPath = args.outfile
    seqPath = args.seqs
    embPath = args.embeddings
    device = args.device
    threshold = args.thresh

    # Set Outpath
    if outPath is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M.predictions")

    logFilePath = outPath + ".log"
    logFile = open(logFilePath,"w+")

    # Set Device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        print(f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}")
        log(f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}", file=logFile)
    else:
        print("# Using CPU")
        log("# Using CPU", file=logFile)

    # Load Model
    try:
        if use_cuda:
            model = torch.load(modelPath).cuda()
        else:
            model = torch.load(modelPath).cpu()
            model.use_cuda = False
    except FileNotFoundError:
        print(f"# Model {modelPath} not found")
        log(f"Model {modelPath} not found", file=logFile)
        logFile.close()
        sys.exit(1)

    # Load Pairs
    try:
        pairs = pd.read_csv(csvPath, sep="\t", header=None)
        all_prots = set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1]))
    except FileNotFoundError:
        print(f"# Pairs File {csvPath} not found")
        log(f"Pairs File {csvPath} not found", file=logFile)
        logFile.close()
        sys.exit(1)

    # Load Sequences or Embeddings
    if embPath is None:
        try:
            names, seqs = parse(open(seqPath, "r"))
            seqDict = {n: s for n, s in zip(names, seqs)}
        except FileNotFoundError:
            print(f"# Sequence File {seqPath} not found")
            log(f"Sequence File {seqPath} not found", file=logFile)
            logFile.close()
            sys.exit(1)
        print("# Generating Embeddings...")
        log("Generating Embeddings...", file=logFile)
        embeddings = {}
        for n in tqdm(all_prots):
            embeddings[n] = lm_embed(seqDict[n], use_cuda)
    else:
        print("# Loading Embeddings...")
        log("Loading Embeddings...", file=logFile)
        embedH5 = h5py.File(embPath, "r")
        embeddings = {}
        for n in tqdm(all_prots):
            embeddings[n] = torch.from_numpy(embedH5[n][:])
        embedH5.close()

    # Make Predictions
    print("# Making Predictions...")
    log("Making Predictions...", file=logFile)
    n = 0
    outPathAll = f"{outPath}.tsv"
    outPathPos = f"{outPath}.positive.tsv"
    cmap_file = h5py.File(f"{outPath}.cmaps.h5", "w")
    model.eval()
    with open(outPathAll, "w+") as f:
        with open(outPathPos, "w+") as pos_f:
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
                        f.write(f"{n0}\t{n1}\t{p}\n")
                        if p >= threshold:
                            pos_f.write(f"{n0}\t{n1}\t{p}\n")
                            cm_np = cm.squeeze().cpu().numpy()
                            dset = cmap_file.require_dataset(f"{n0}x{n1}", cm_np.shape, np.float32)
                            dset[:] = cm_np
                            #cmap_file.create_dataset(f"{n0}x{n1}", data=cm.squeeze().cpu().numpy())
                    except RuntimeError as e:
                        log(f"{n0} x {n1} skipped - CUDA out of memory", file=logFile)

    logFile.close()
    cmap_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
