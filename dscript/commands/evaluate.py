"""
Evaluate a trained model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import h5py
import datetime
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)
from tqdm import tqdm

matplotlib.use("Agg")


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
    parser.add_argument("-o", "--outfile", help="Output file to write results")
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    return parser


def plot_eval_predictions(labels, predictions, path="figure"):
    """
    Plot histogram of positive and negative predictions, precision-recall curve, and receiver operating characteristic curve.

    :param y: Labels
    :type y: np.ndarray
    :param phat: Predicted probabilities
    :type phat: np.ndarray
    :param path: File prefix for plots to be saved to [default: figure]
    :type path: str
    """

    pos_phat = predictions[labels == 1]
    neg_phat = predictions[labels == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Distribution of Predictions")
    ax1.hist(pos_phat)
    ax1.set_xlim(0, 1)
    ax1.set_title("Positive")
    ax1.set_xlabel("p-hat")
    ax2.hist(neg_phat)
    ax2.set_xlim(0, 1)
    ax2.set_title("Negative")
    ax2.set_xlabel("p-hat")
    plt.savefig(path + ".phat_dist.png")
    plt.close()

    precision, recall, pr_thresh = precision_recall_curve(labels, predictions)
    aupr = average_precision_score(labels, predictions)
    print("AUPR:", aupr)

    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall (AUPR: {:.3})".format(aupr))
    plt.savefig(path + ".aupr.png")
    plt.close()

    fpr, tpr, roc_thresh = roc_curve(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    print("AUROC:", auroc)

    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Receiver Operating Characteristic (AUROC: {:.3})".format(auroc))
    plt.savefig(path + ".auroc.png")
    plt.close()


def main(args):
    """
    Run model evaluation from arguments.

    :meta private:
    """

    # Set Device
    device = args.device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        print(
            f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}"
        )
    else:
        print("# Using CPU")

    # Load Model
    model_path = args.model
    if use_cuda:
        model = torch.load(model_path).cuda()
    else:
        model = torch.load(model_path).cpu()
        model.use_cuda = False

    embeddingPath = args.embedding
    h5fi = h5py.File(embeddingPath, "r")

    # Load Pairs
    test_fi = args.test
    test_df = pd.read_csv(test_fi, sep="\t", header=None)

    if args.outfile is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    else:
        outPath = args.outfile
    outFile = open(outPath + ".predictions.tsv", "w+")

    allProteins = set(test_df[0]).union(test_df[1])

    seqEmbDict = {}
    for i in tqdm(allProteins, desc="Loading embeddings"):
        seqEmbDict[i] = torch.from_numpy(h5fi[i][:]).float()

    model.eval()
    with torch.no_grad():
        phats = []
        labels = []
        for _, (n0, n1, label) in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Predicting pairs"
        ):
            try:
                p0 = seqEmbDict[n0]
                p1 = seqEmbDict[n1]
                if use_cuda:
                    p0 = p0.cuda()
                    p1 = p1.cuda()

                pred = model.predict(p0, p1).item()
                phats.append(pred)
                labels.append(label)
                print(
                    "{}\t{}\t{}\t{:.5}".format(n0, n1, label, pred),
                    file=outFile,
                )
            except Exception as e:
                sys.stderr.write("{} x {} - {}".format(n0, n1, e))

    phats = np.array(phats)
    labels = np.array(labels)
    plot_eval_predictions(labels, phats, outPath)

    outFile.close()
    h5fi.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
