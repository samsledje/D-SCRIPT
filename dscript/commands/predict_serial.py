"""
Make new predictions with a pre-trained model using legacy (serial) inference. One of --seqs or --embeddings is required.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Callable
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..fasta import parse
from ..foldseek import fold_vocab, get_foldseek_onehot
from ..language_model import lm_embed
from ..models.interaction import DSCRIPTModel
from ..utils import load_hdf5_parallel, log


class PredictionArguments(NamedTuple):
    cmd: str
    device: int
    embeddings: str | None
    outfile: str | None
    seqs: str
    model: str | None
    thresh: float | None
    load_proc: int | None
    func: Callable[[PredictionArguments], None]


def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """

    parser.add_argument(
        "--pairs", help="Candidate protein pairs to predict", required=True
    )
    parser.add_argument(
        "--model",
        default="samsl/topsy_turvy_human_v1",
        type=str,
        help="Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_human_v1]",
    )
    parser.add_argument("--seqs", help="Protein sequences in .fasta format")
    parser.add_argument("--embeddings", help="h5 file with embedded sequences")
    parser.add_argument(
        "--foldseek_fasta",
        help="""3di sequences in .fasta format. Can be generated using `dscript extract-3di.
        Default is None. If provided, TT3D will be run, otherwise default D-SCRIPT/TT will be run.
        """,
        default=None,
    )
    parser.add_argument("-o", "--outfile", help="File for predictions")
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    parser.add_argument(
        "--store_cmaps",
        action="store_true",
        help="Store contact maps for predicted pairs above `--thresh` in an h5 file",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.5,
        help="Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]",
    )
    parser.add_argument(
        "--load_proc",
        type=int,
        default=32,
        help="Number of processes to use when loading embeddings (-1 = # of CPUs, default=32)",
    )
    return parser


def main(args):
    """
    Run new prediction from arguments.

    :meta private:
    """
    if args.seqs is None and args.embeddings is None:
        log("One of --seqs or --embeddings is required.")
        sys.exit(0)

    tsvPath = args.pairs
    modelPath = args.model
    outPath = args.outfile
    seqPath = args.seqs
    embPath = args.embeddings
    device = args.device
    threshold = args.thresh

    foldseek_fasta = args.foldseek_fasta

    # Set Outpath
    if outPath is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M.predictions")

    logFilePath = outPath + ".log"
    logFile = open(logFilePath, "w+")

    # Set Device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=logFile,
            print_also=True,
        )
    else:
        log("Using CPU", file=logFile, print_also=True)

    # Load Model
    log(f"Loading model from {modelPath}", file=logFile, print_also=True)
    if modelPath.endswith(".sav") or modelPath.endswith(".pt"):
        try:
            if use_cuda:
                model = torch.load(modelPath).cuda()
                model.use_cuda = True
            else:
                model = torch.load(
                    modelPath,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                ).cpu()
                model.use_cuda = False
        except FileNotFoundError:
            log(f"Model {modelPath} not found", file=logFile, print_also=True)
            logFile.close()
            sys.exit(1)
    else:
        try:
            model = DSCRIPTModel.from_pretrained(modelPath, use_cuda=use_cuda)
            if use_cuda:
                model = model.cuda()
                model.use_cuda = True
            else:
                model = model.cpu()
                model.use_cuda = False
        except Exception as e:
            print(e)
            log(f"Model {modelPath} failed: {e}", file=logFile, print_also=True)
            logFile.close()
            sys.exit(1)
    if (
        dict(model.named_parameters())["contact.hidden.conv.weight"].shape[1] == 242
    ) and (foldseek_fasta is None):
        raise ValueError(
            "A TT3D model has been provided, but no foldseek_fasta has been provided"
        )

    # Load Pairs
    try:
        log(f"Loading pairs from {tsvPath}", file=logFile, print_also=True)
        pairs = pd.read_csv(tsvPath, sep="\t", header=None)
        all_prots = set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1]))
    except FileNotFoundError:
        log(f"Pairs File {tsvPath} not found", file=logFile, print_also=True)
        logFile.close()
        sys.exit(1)

    # Load Sequences or Embeddings
    torch.multiprocessing.set_sharing_strategy("file_system")
    if embPath is None:
        try:
            names, seqs = parse(seqPath, "r")
            seqDict = {n: s for n, s in zip(names, seqs, strict=False)}
        except FileNotFoundError:
            log(
                f"Sequence File {seqPath} not found",
                file=logFile,
                print_also=True,
            )
            logFile.close()
            sys.exit(1)
        log("Generating Embeddings...", file=logFile, print_also=True)
        embeddings = {}
        for n in tqdm(all_prots):
            embeddings[n] = lm_embed(seqDict[n], use_cuda)
    else:
        log("Loading Embeddings...", file=logFile, print_also=True)
        embeddings = load_hdf5_parallel(
            embPath, all_prots, n_jobs=args.load_proc
        )  # Is a dict, legacy behavior

    # Load Foldseek Sequences
    if foldseek_fasta is not None:
        log("Loading FoldSeek 3Di sequences...", file=logFile, print_also=True)
        try:
            fs_names, fs_seqs = parse(foldseek_fasta, "r")
            fsDict = {n: s for n, s in zip(fs_names, fs_seqs, strict=False)}
        except FileNotFoundError:
            log(
                f"Foldseek Sequence File {foldseek_fasta} not found",
                file=logFile,
                print_also=True,
            )
            logFile.close()
            sys.exit(1)

    # Make Predictions
    log("Making Predictions...", file=logFile, print_also=True)
    n = 0
    outPathAll = f"{outPath}.tsv"
    outPathPos = f"{outPath}.positive.tsv"
    if args.store_cmaps:
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

                    # Load foldseek one-hot
                    if foldseek_fasta is not None:
                        fs0 = get_foldseek_onehot(
                            n0, p0.shape[1], fsDict, fold_vocab
                        ).unsqueeze(0)
                        fs1 = get_foldseek_onehot(
                            n1, p1.shape[1], fsDict, fold_vocab
                        ).unsqueeze(0)
                        if use_cuda:
                            fs0 = fs0.cuda()
                            fs1 = fs1.cuda()

                    try:
                        if foldseek_fasta is not None:
                            try:
                                cm, p = model.map_predict(p0, p1, True, fs0, fs1)
                            except TypeError as e:
                                log(e)
                                log(
                                    "Loaded model does not support foldseek. Please retrain with --allow_foldseek or download a pre-trained TT3D model."
                                )
                                raise e
                        else:
                            cm, p = model.map_predict(p0, p1)
                        p = p.item()
                        f.write(f"{n0}\t{n1}\t{p}\n")
                        if p >= threshold:
                            pos_f.write(f"{n0}\t{n1}\t{p}\n")
                            if args.store_cmaps:
                                cm_np = cm.squeeze().cpu().numpy()
                                dset = cmap_file.require_dataset(
                                    f"{n0}x{n1}", cm_np.shape, np.float32
                                )
                                dset[:] = cm_np
                    except RuntimeError as e:
                        log(
                            f"{n0} x {n1} skipped ({e})",
                            file=logFile,
                        )

    logFile.close()
    if args.store_cmaps:
        cmap_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
