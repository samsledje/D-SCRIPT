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
import os, math

import torch.multiprocessing as mp

from ..fasta import parse
from ..foldseek import get_foldseek_onehot, fold_vocab
from ..language_model import lm_embed
from ..utils import log
from ..loading import LoadingPool


#When a new process is started with spawn, the file containing the target function will be passed
#So, the function should be in its own file to minimize the cost and remove any risk.
from .par_worker import _predict
from .par_writer import _writer

class BlockedPredictionArguments(NamedTuple):
    cmd: str
    device: int
    embeddings: str
    #foldseek_fasta: Optional[str] - missing from original class for some reason?
    outfile: Optional[str]
    model: str
    thresh: Optional[float]
    load_proc: Optional[int]
    blocks: Optional[int]
    func: Callable[[PredictionArguments], None]


def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """

    parser.add_argument(
        "--proteins", help="Protein IDs for which to predict all pairs", required=True
    )
    parser.add_argument("--model", help="Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_v1]")
    #parser.add_argument("--seqs", help="Protein sequences in .fasta format")
    parser.add_argument("--embeddings", help="h5 file with embedded sequences", required=True
    )
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
        default=-1,
        help="Number of processes to use when loading embeddings (-1 = # of CPUs)"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=16,
        help="Number of equal-sized blocks to split proteins into. Maximum (embedding) memory usage should be 3 blocks worth"
    )
    return parser
    
def main(args):
    """
    Run new prediction from arguments.

    :meta private:
    """
    if args.proteins is None or args.embeddings is None:
        log("Both of --proteins and --embeddings are required for blocked prediction.")
        sys.exit(0)

    csvPath = args.proteins
    modelPath = args.model
    outPath = args.outfile
    #seqPath = args.seqs
    embPath = args.embeddings
    device = args.device
    threshold = args.thresh

    foldseek_fasta = args.foldseek_fasta
    num_blocks = args.blocks

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
        log(f"Loading protein IDs from {csvPath}", file=logFile, print_also=True)
        with open(csvPath) as f:
            all_prots = [line.strip() for line in f if line and not line.isspace()]
            #TODO: make slices alias, not copy
    except FileNotFoundError:
        log(f"Pairs File {csvPath} not found", file=logFile, print_also=True)
        logFile.close()
        sys.exit(1)

    #prot_to_idx = {p:i for i,p in enumerate(all_prots)} #Name -> index

    # Check Embeddings File
    if not os.path.exists(embPath):
        log(f"Embeddings File {embPath} not found. Pre-computed embeddings are required to use blocked prediction.", file=logFile, print_also=True)
        logFile.close()
        sys.exit(1)
    #TODO: Pre-embedding, as it is now, actually requires holding all embeddings in memory
    #So, maybe we want to modify that or allow on the fly (but, this would re-emebd the same things...)
    #embeddings = load_hdf5_parallel(embPath, all_prots, n_jobs=args.load_proc) #Note: this is now a list

    # Load Foldseek Sequences
    use_fs = foldseek_fasta is not None
    if use_fs and not os.path.exists(foldseek_fasta):
        log(f"Foldseek FASTA File {foldseek_fasta} specified but not found.", file=logFile, print_also=True)
        logFile.close()
        sys.exit(1)
    #fs_names, fs_seqs = parse(foldseek_fasta, "r") #Not parallel
    #fsDict = {n:s for n, s in zip(fs_names, fs_seqs)}
    #fsOnehotList = [None]*len(all_prots)
    #use_fs = True


    # Make Predictions
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    n_prots = len(all_prots)
    n_pairs = int(n_prots * (n_prots - 1) / 2) #n choose 2
    #n_gpu = torch.cuda.device_count()

    #This uses the pytorch spawn function to start a bunch of processes using spawn
    #Apparently, spawn is required when using CUDA in the processes
    #proc_ctx = mp.spawn(_predict, 
    #                    args=(modelPath, input_queue, output_queue, args.store_cmaps, use_fs, None), #Can't pass an open file
    #                    nprocs=n_gpu, join=False)
    p = mp.Process(target=_predict, args=(device, modelPath, input_queue, output_queue, args.store_cmaps, use_fs, None))
    p.start()
    
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

    #Create the pool we will use to load embeddings
    loadpool = LoadingPool(embPath, n_jobs=args.load_proc)

    block_size = math.ceil(n_prots / num_blocks)

    #We move through pairs of blocks in the order (0,0), (0,1), ..., (0,n), (1,n), ..., (1,3), (1,1), (1,2), (2,2), ...

    def get_bounds(block): 
        start = block * block_size
        end = min(start+block_size, n_prots)
        return (start, end) #all_prots[start:end]

    def load_prots(block): #TODO: make this alias, not copy
        return loadpool.load(all_prots[slice(*get_bounds(block))])
    
    def submit_self_block(block, embeddings):
        start, end = get_bounds(block)
        for i0 in range(start, end):
            p0 = embeddings[i0-start]
            for i1 in range(i0+1, end):
                p1 = embeddings[i1-start]
                #TODO - re-enable foldseek, I guess? Ugh
                #if use_fs:
                #    if fsOnehotList[i0]: 
                #        fs0 = fsOnehotList[i0]
                #    else:
                #        fs0 = get_foldseek_onehot(n0, p0.shape[1], fsDict, fold_vocab).unsqueeze(0)
                #        fs0.share_memory_()
                #        fsOnehotList[i0] = fs0
                #    if i1 in fsOnehotList:
                #        fs1 = fsOnehotList[i1]
                #    else:
                #        fs1 = get_foldseek_onehot(n1, p1.shape[1], fsDict, fold_vocab).unsqueeze(0)
                #        fs1.share_memory_()
                #        fsOnehotList[i1] = fs1
                #    tup = (i0,i1,p0,p1,fs0,fs1)
                #else:
                tup = (i0,i1,p0,p1)
                input_queue.put(tup) 
        print("Self-block submitted", block)

    def submit_block(block1, block2, embeddings1, embeddings2, flag=True):
        start1, end1 = get_bounds(block1)
        start2, end2 = get_bounds(block2)
        for i0 in range(start1, end1):
            p0 = embeddings1[i0-start1]
            for i1 in range(start2, end2):
                p1 = embeddings2[i1-start2]
                #TODO - re-enable foldseek, I guess? Ugh
                #if use_fs:
                #    if fsOnehotList[i0]: 
                #        fs0 = fsOnehotList[i0]
                #    else:
                #        fs0 = get_foldseek_onehot(n0, p0.shape[1], fsDict, fold_vocab).unsqueeze(0)
                #        fs0.share_memory_()
                #        fsOnehotList[i0] = fs0
                #    if i1 in fsOnehotList:
                #        fs1 = fsOnehotList[i1]
                #    else:
                #        fs1 = get_foldseek_onehot(n1, p1.shape[1], fsDict, fold_vocab).unsqueeze(0)
                #        fs1.share_memory_()
                #        fsOnehotList[i1] = fs1
                #    tup = (i0,i1,p0,p1,fs0,fs1)
                #else:
                tup = (i0,i1,p0,p1)
                input_queue.put(tup) 
        #TODO: specify block-end flag if flag
        print("Block submitted", block1, block2)

   
    data1 = load_prots(0)
    data2 = None
    data3 = None
    #TODO: set up block-done-queue, put an imaginary 0 in it...
    for i in range(num_blocks):
        if i % 2 == 0:
            #Do self block
            submit_self_block(i, data1)
            if i == 0 and num_blocks > 1:
                data3 = load_prots(1)
            for j in range(i+1, num_blocks):
                if j == i+1:
                    data2 = data3
                else:
                    data2 = load_prots(j) #TODO: wait here
                submit_block(i, j, data1, data2)
        else:
            data1 = load_prots(i) if i < num_blocks - 1 else data2 #TODO: wait here if loading
            for j in range(num_blocks-1, i+2, -1):
                submit_block(i, j, data1, data2)
                data2 = load_prots(j-1)# TODO: wait here
            if i < num_blocks - 2:
                submit_block(i, i+2, data1, data2, flag=False)
                data3 = data2
            #do self-block
            submit_self_block(i, data1)
            if i < num_blocks-1:
                if i != num_blocks-2:
                    data2 = load_prots(i+1) #TODO: wait here
                submit_block(i, i+1, data1, data2)
            data1 = data2
    
    loadpool.shutdown()
    input_queue.put(None)
    write_proc.join()

    log(f"All predictions completed", file=logFile, print_also=True)
    logFile.close() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
