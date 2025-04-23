"""
Make new predictions with a pre-trained model. One of --seqs or --embeddings is required.
"""
from __future__ import annotations
import argparse
import datetime
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Callable, NamedTuple, Optional

import torch.multiprocessing as mp

from ..fasta import parse
from ..foldseek import get_foldseek_onehot, fold_vocab
from ..language_model import lm_embed
from ..utils import log, load_hdf5_parallel

#When a new process is started with spawn, the file containing the target function will be passed
#So, the function should be in its own file to minimize the cost and remove any risk.
from .par_worker import _predict
from .par_writer import _writer

def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """

    parser.add_argument(
        "--pairs", help="Candidate protein pairs to predict", required=True
    )
    parser.add_argument("--model", help="Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_v1]")
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
    #parser.add_argument(
    #    "-d", "--device", type=int, default=-1, help="Compute device to use"
    #)
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
        default=-1,
        help="Number of processes to use when loading embeddings (-1 = # of CPUs)"
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

    csvPath = args.pairs
    modelPath = args.model
    outPath = args.outfile
    seqPath = args.seqs
    embPath = args.embeddings
    #device = args.device
    threshold = args.thresh

    foldseek_fasta = args.foldseek_fasta

    # Set Outpath
    if outPath is None:
        outPath = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H:%M.predictions"
        )

    logFilePath = outPath + ".log"
    logFile = open(logFilePath, "w+")

    # Set Device
    assert torch.cuda.is_available()

    #CUDA-using processes need to be spawned; and, the start method needs to be
    # #set before the queues are created so they match the processes
    mp.set_start_method('spawn')
    #For torch shared memory
    mp.set_sharing_strategy("file_system")

    # Load Pairs
    try:
        log(f"Loading pairs from {csvPath}", file=logFile, print_also=True)
        pairs = pd.read_csv(csvPath, sep="\t", header=None)
        all_prots = list(set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1])))
    except FileNotFoundError:
        log(f"Pairs File {csvPath} not found", file=logFile, print_also=True)
        logFile.close()
        sys.exit(1)

    prot_to_idx = {p:i for i,p in enumerate(all_prots)} #Name -> index

    # Load Sequences or Embeddings
    if embPath is None:
        try:
            names, seqs = parse(seqPath, "r")
            seqDict = {n: s for n, s in zip(names, seqs)}
        except FileNotFoundError:
            log(
                f"Sequence File {seqPath} not found",
                file=logFile,
                print_also=True,
            )
            logFile.close()
            sys.exit(1)
        log("Generating Embeddings...", file=logFile, print_also=True)
        embeddings = [None] * len(all_prots)
        for i,n in tqdm(enumerate(all_prots)):
            emb = lm_embed(seqDict[n], True) #This is not parallel right now
            emb.share_memory_()
            embeddings[i] = emb #Could also just append as we go, since in sequential mode.
    else:
        log("Loading Embeddings...", file=logFile, print_also=True)
        embeddings = load_hdf5_parallel(embPath, prot_to_idx, n_jobs=args.load_proc) #Note: this is now a list

    # Load Foldseek Sequences
    if foldseek_fasta is not None:
        log("Loading FoldSeek 3Di sequences...", file=logFile, print_also=True)
        try:
            fs_names, fs_seqs = parse(foldseek_fasta, "r") #Not parallel
            fsDict = {n:s for n, s in zip(fs_names, fs_seqs)}
            fsOnehotList = [None]*len(all_prots)
            use_fs = True
        except FileNotFoundError:
            log(
                f"Foldseek Sequence File {foldseek_fasta} not found",
                file=logFile,
                print_also=True,
            )
            logFile.close()
            sys.exit(1)
    else:
        use_fs = False

    # Make Predictions
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    n_pairs = len(pairs)
    n_gpu = torch.cuda.device_count()

    #This uses the pytorch spawn function to start a bunch of processes using spawn
    #Apparently, spawn is required when using CUDA in the processes
    proc_ctx = mp.spawn(_predict, 
                        args=(modelPath, input_queue, output_queue, args.store_cmaps, use_fs, None), #Can't pass an open file
                        nprocs=n_gpu, join=False)
    #for i in range(n_gpu):
    #    p = mp.Process(target=predict, args=(i, modelPath, input_queue, output_queue, args.store_cmaps, use_fs, logFile if i == 0 else None)) #Only the first process gets to log
    #    p.start()
    
    outPathAll = f"{outPath}.tsv"
    outPathPos = f"{outPath}.positive.tsv"
    if args.store_cmaps:
        cmapPath = f"{outPath}.cmaps.h5"
    else:
        cmapPath = None

    #The writer needs to be seperate from the main process so that writing can start before all pairs are in the queue
    #We are still passing a large array (list of seq names) but it saves us passing names of all pairs of strings
    #Note that we can't share a queue between spawned and forked processes, so this also needs to be spawned
    write_proc = mp.Process(target=_writer, args=(all_prots, outPathAll, outPathPos, cmapPath, n_pairs, threshold, output_queue))
    write_proc.start()


    for _, (n0, n1) in pairs.iloc[:, :2].iterrows():
        n0 = str(n0)
        n1 = str(n1)
        i0 = prot_to_idx[n0]
        i1 = prot_to_idx[n1]
        p0 = embeddings[i0]
        p1 = embeddings[i1]
        #Previously, fs onehot tuples were generated on the fly. This switches to generating them in advanced, allowing us to
        #share pointers and not need to pass fsDict. It also let's us reuse computation.
        #I'm not sure that all sequences are used, so they are computed as sequences are encountered
        if use_fs:
            if fsOnehotList[i0]: 
                fs0 = fsOnehotList[i0]
            else:
                fs0 = get_foldseek_onehot(n0, p0.shape[1], fsDict, fold_vocab).unsqueeze(0)
                fs0.share_memory_()
                fsOnehotList[i0] = fs0
            if i1 in fsOnehotList:
                fs1 = fsOnehotList[i1]
            else:
                fs1 = get_foldseek_onehot(n1, p1.shape[1], fsDict, fold_vocab).unsqueeze(0)
                fs1.share_memory_()
                fsOnehotList[i1] = fs1
            tup = (i0,i1,p0,p1,fs0,fs1)
        else:
            tup = (i0,i1,p0,p1)
        input_queue.put(tup) 
    # Signal workers to stop after processing all tasks
    for _ in range(n_gpu):
        input_queue.put(None)
    #The writing process knows how many pairs to expect, so we only need to wait for it to finish.    
    write_proc.join()

    log(f"All predictions completed", file=logFile, print_also=True)
    logFile.close() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
