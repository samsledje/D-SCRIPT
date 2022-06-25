"""
Make new predictions with a pre-trained model. One of --seqs or --embeddings is required.
"""
import argparse
import datetime
import logging as logg
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.special import comb
from tqdm import tqdm

from ..datamodules import CachedFasta, CachedH5
from ..alphabets import Uniprot21
from ..fasta import parse
from ..language_model import lm_embed
from ..utils import load_hdf5_parallel


def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """

    parser.add_argument(
        "--pairs", help="Candidate protein pairs to predict", required=True
    )
    parser.add_argument("--model", help="Pretrained Model", required=True)
    parser.add_argument("--seqs", help="Protein sequences in .fasta format")
    parser.add_argument("--embeddings", help="h5 file with embedded sequences")
    parser.add_argument(
        "--preload",
        type=bool,
        default=False,
        help="h5 file with embedded sequences",
    )
    parser.add_argument("-o", "--outfile", help="File for predictions")
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
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
    # Set Outpath
    outPath = args.outfile
    if outPath is None:
        outPath = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M.predictions"
        )

    logFilePath = outPath + ".log"
    logg.basicConfig(
        level=logg.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logg.FileHandler(logFilePath),
            logg.StreamHandler(sys.stdout),
        ],
    )

    if args.seqs is None and args.embeddings is None:
        logg.error("One of --seqs or --embeddings is required.")
        sys.exit(1)

    csvPath = args.pairs
    modelPath = args.model
    seqPath = args.seqs
    embPath = args.embeddings
    device = args.device
    threshold = args.thresh
    preload = args.preload

    # Set Device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        logg.info(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}"
        )
    else:
        logg.info("Using CPU")

    # Load Model
    try:
        model = torch.load(modelPath).eval()
    except FileNotFoundError:
        logg.error(f"Model {modelPath} not found")
        sys.exit(1)
    if use_cuda:
        model = model.cuda()
        model.use_cuda = True
    else:
        model = model.cpu()
        model.use_cuda = False

    # Load Pairs
    try:
        logg.info(f"Loading pairs from {modelPath}")
        pairs = pd.read_csv(csvPath, sep="\t", header=None)
        all_prots = set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1]))
    except FileNotFoundError:
        logg.error(f"Pairs File {csvPath} not found")
        sys.exit(1)

    if pairs.shape[1] > 2:
        logg.error(f"Pairs file should have two columns (has {pairs.shape[1]}")
        sys.exit(1)

    # Load Sequences or Embeddings
    if embPath is None:
        try:
            embeddings = CachedFasta(seqPath, preload)
        except FileNotFoundError:
            logg.error(f"Sequence File {seqPath} not found")
            sys.exit(1)
    else:
        embeddings = CachedH5(embPath, preload)

    if all_prots.difference(embeddings.seqs):
        logg.error(
            "Sequences requested in pairs file not present in sequence file."
        )
        logg.debug(all_prots.difference(embeddings.seqs))
        logg.debug(list(embeddings.seqMap.keys()))
        sys.exit(1)

    # Make Predictions
    logg.info("Making Predictions...")
    outPathAll = f"{outPath}.tsv"
    outPathPos = f"{outPath}.positive.tsv"

    with open(outPathAll, "w+") as out_f, open(
        outPathPos, "w+"
    ) as pos_f, h5py.File(
        f"{outPath}.cmaps.h5", "w"
    ) as cmap_file, torch.no_grad():
        for i, (n0, n1) in tqdm(pairs.iterrows(), total=len(pairs)):
            if i % 50 == 0:
                out_f.flush()
            p0 = embeddings[n0]
            p1 = embeddings[n1]
            if use_cuda:
                p0 = p0.cuda()
                p1 = p1.cuda()
            try:
                cm, p = model.map_predict(p0, p1)
                p = p.item()
                out_f.write(f"{n0}\t{n1}\t{p}\n")
                if p >= threshold:
                    pos_f.write(f"{n0}\t{n1}\t{p}\n")
                    cm_np = cm.squeeze().cpu().numpy()
                    dset = cmap_file.require_dataset(
                        f"{n0}x{n1}", cm_np.shape, np.float32
                    )
                    dset[:] = cm_np
            except RuntimeError as e:
                logg.warning(e)
                logg.warning(f"{n0} x {n1} skipped - CUDA out of memory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
