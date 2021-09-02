"""
Evaluate a trained model.
"""

import argparse
import datetime
import logging as logg
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..datamodules import CachedH5
from ..utils import plot_eval_predictions


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """

    parser.add_argument(
        "--model", help="Trained prediction model", required=True
    )
    parser.add_argument("--test", help="Test Data", required=True)
    parser.add_argument(
        "--embedding", help="h5 file with embedded sequences", required=True
    )
    parser.add_argument(
        "--preload",
        type=bool,
        default=False,
        help="h5 file with embedded sequences",
    )
    parser.add_argument("-o", "--outfile", help="Output file to write results")
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    return parser


def main(args):
    """
    Run model evaluation from arguments.

    :meta private:
    """

    if args.outfile is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    else:
        outPath = args.outfile
    logPath = f"{outPath}.log"

    logg.basicConfig(
        level=logg.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logg.FileHandler(logPath), logg.StreamHandler(sys.stdout)],
    )

    # Set Device
    device = args.device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        logg.info(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}"
        )
    else:
        print("Using CPU")

    # Load Model
    logg.info("Loading model...")
    modelPath = args.model
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
    model.eval()

    # Load Embeddings
    logg.info("Loading embeddings...")
    embPath = args.embedding
    preload = args.preload
    embeddings = CachedH5(embPath, preload)

    # Load Pairs
    logg.info("Loading pairs...")
    test_fi = args.test
    try:
        test_df = pd.read_csv(test_fi, sep="\t", header=None)
        all_prots = set(test_df.iloc[:, 0]).union(set(test_df.iloc[:, 1]))
    except FileNotFoundError:
        logg.error(f"Pairs File {test_fi} not found")
        sys.exit(1)

    if test_df.shape[1] != 3:
        logg.error(
            f"Pairs file should have three columns (has {test_df.shape[1]}"
        )
        sys.exit(1)

    if all_prots.difference(embeddings.seqs):
        logg.error(
            "Sequences requested in pairs file not present in sequence file."
        )
        logg.debug(all_prots.difference(embeddings.seqs))
        logg.debug(list(embeddings.seqMap.keys()))
        sys.exit(1)

    logg.info("Beginning evaluation...")
    with open(f"{outPath}.evaluation.tsv", "w+") as out_f, torch.no_grad():
        phats = []
        labels = []
        for i, (n0, n1, label) in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Predicting pairs"
        ):
            if i % 50 == 0:
                out_f.flush()
            try:
                p0 = embeddings[n0]
                p1 = embeddings[n1]
                if use_cuda:
                    p0 = p0.cuda()
                    p1 = p1.cuda()
                pred = model.predict(p0, p1).item()
                phats.append(pred)
                labels.append(label)
                out_f.write(
                    "{}\t{}\t{}\t{:.5}".format(n0, n1, label, pred),
                )
            except Exception as e:
                logg.error("{} x {} - {}".format(n0, n1, e))

    phats = np.array(phats)
    labels = np.array(labels)
    plot_eval_predictions(labels, phats, outPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
