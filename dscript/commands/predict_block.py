"""
Make new predictions with a pre-trained model using blocked, multi-GPU pariwise inference. One of --proteins and --pairs is required.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import torch
import torch.multiprocessing as mp

from ..fasta import parse_from_list
from ..foldseek import fold_vocab, get_foldseek_onehot
from ..loading import LoadingPool
from ..utils import log, parse_device

# When a new process is started with spawn, the file containing the target function will be passed
# So, the function should be in its own file to minimize the cost and remove any risk.
from .par_worker import _predict
from .par_writer import _writer


class BlockedPredictionArguments(NamedTuple):
    cmd: str
    protins: str | None
    pairs: str | None
    model: str | None
    embeddings: str
    foldseek_fasta: str | None
    outfile: str | None
    device: str | None
    thresh: float | None
    load_proc: int | None
    blocks: int | None
    func: Callable[[BlockedPredictionArguments], None]


def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """

    parser.add_argument(
        "--proteins",
        help="File with protein IDs for which to predict all pairs, one per line; specify one of proteins or pairs",
        required=False,
    )
    parser.add_argument(
        "--pairs",
        help="File with candidate protein pairs to predict, one pair per line; specify one of proteins or pairs",
        required=False,
    )
    parser.add_argument(
        "--model",
        help="Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_human_v1]",
        default="samsl/topsy_turvy_human_v1",
    )
    parser.add_argument(
        "--embeddings",
        help="h5 file with (a superset of) pre-embedded sequences. Generate with dscript embed.",
        required=True,
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
        "-d",
        "--device",
        type=str,
        default="all",
        help="Compute device to use. Options: 'cpu', 'all' (all GPUs), or GPU index (0, 1, 2, etc.). To use specific GPUs, set CUDA_VISIBLE_DEVICES beforehand and use 'all'. [default: all]",
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
        default=16,
        help="Number of processes to use when loading embeddings (-1 = # of available CPUs, default=16). Because loading is IO-bound, values larger that the # of CPUs are allowed.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="Number of equal-sized blocks to split proteins into. In the multi-block case, maximum (embedding) memory usage should be 3 blocks' worth. When multiple GPUs are used, memory usage may briefly be higher when different GPUs are working on tasks from different blocks. And, small blocks may lead to occasional brief hangs with multiple GPUs. Default 1.",
    )
    parser.add_argument(
        "--sparse_loading",
        action="store_true",
        help="Load only the proteins required from each block, but do not reuse loaded blocks in memory. Recommented when predicting with many blocks on sparse pairs, such that many pairs of blocks might contain no pairs of proteins of interest. Only available when blocks > 1 and pairs specified. Maximum (embedding) memory usage with this option is 4 blocks' worth.",
    )
    return parser


def main(args):
    """
    Run new prediction from arguments.

    :meta private:
    """

    # Deal with provided arguments
    # Set Outpath
    outPath = args.outfile
    if outPath is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M.predictions")

    logFilePath = outPath + ".log"
    logFile = open(logFilePath, "w+")

    if args.proteins is None == args.pairs is None:
        log(
            "Please specify exactly one of proteins and pairs.",
            file=logFile,
            print_also=True,
        )
        logFile.close()
        sys.exit(2)

    embPath = args.embeddings
    if not os.path.exists(embPath):
        log(f"Embeddings File {embPath} not found.", file=logFile, print_also=True)
        logFile.close()
        sys.exit(3)

    modelPath = args.model
    device = parse_device(args.device, logFile)

    threshold = args.thresh
    foldseek_fasta = args.foldseek_fasta
    num_blocks = args.blocks

    # Check model path
    if modelPath.endswith(".sav") or modelPath.endswith(".pt"):
        if os.path.isfile(modelPath):
            log(
                f"Will load model locally from {modelPath}", file=logFile, print_also=True
            )
        else:
            log(f"Local model {modelPath} not found", file=logFile, print_also=True)
            logFile.close()
            sys.exit(6)
    else:
        log(
            f"Will attempt to download HuggingFace model from {modelPath}",
            file=logFile,
            print_also=True,
        )

    # CUDA-using processes need to be spawned; and, the start method needs to be
    # #set before the queues are created so they match the processes
    mp.set_start_method("spawn")
    # For torch shared memory
    mp.set_sharing_strategy("file_system")

    # Load Proteins
    all_pairs = args.proteins is not None
    if all_pairs:
        tsvPath = args.proteins
    elif args.pairs is not None:
        tsvPath = args.pairs
    else:
        log(
            "One of --proteins and --pairs must be specified.",
            file=logFile,
            print_also=True,
        )
        logFile.close()
        sys.exit(4)
    try:
        log(
            f"Loading {'' if all_pairs else 'pairs of '}protein IDs from {tsvPath}",
            file=logFile,
            print_also=True,
        )
        with open(tsvPath) as f:
            tsv_lines = [line.strip() for line in f if line and not line.isspace()]
    except FileNotFoundError:
        log(f"Proteins / Pairs file {tsvPath} not found", file=logFile, print_also=True)
        logFile.close()
        sys.exit(4)

    if all_pairs:
        all_prots = tsv_lines
        n_prots = len(all_prots)
        n_pairs = int(n_prots * (n_prots - 1) / 2)  # n choose 2
    # Process a list of pairs into a binary matrix. Not asymptotically efficient for sparse pairs.
    else:
        # Built the data structures we need jointly
        # Also, preserve order of proteins in order of first encounter
        pairs0 = []
        pairs1 = []
        all_prots = []
        prot_to_idx = {}
        for pair in tsv_lines:
            p0, p1 = pair.split("\t")[:2]
            if p0 in prot_to_idx:
                i0 = prot_to_idx[p0]
            else:
                i0 = len(all_prots)
                prot_to_idx[p0] = i0
                all_prots.append(p0)
            if p1 in prot_to_idx:
                i1 = prot_to_idx[p1]
            else:
                i1 = len(all_prots)
                prot_to_idx[p1] = i1
                all_prots.append(p1)
            pairs0.append(i0)
            pairs1.append(i1)
        # Alternative code to construct similar data structures:
        # pairs = [tuple(pair.split()[:2]) for pair in all_prots]
        # all_prots = list(set([prot for pair in pairs for prot in pair]))
        # prot_to_idx = {p:i for i,p in enumerate(all_prots)} #Name -> index

        n_prots = len(all_prots)
        n_pairs = len(pairs0)

        # Need to do this later because we don't know the # of unique proteins
        # We could make this more efficint by keeping only the upper triangular part of the matrix in sparse storage
        pairs_bool = np.zeros((n_prots, n_prots), dtype=np.bool_)
        pairs_bool[pairs0, pairs1] = 1
        pairs_bool[pairs1, pairs0] = 1
        pairs_bool = np.triu(pairs_bool)  # Makes a copy

    # Check Foldseek Sequence File
    use_fs = foldseek_fasta is not None
    if use_fs and not os.path.exists(foldseek_fasta):
        log(
            f"Foldseek FASTA File {foldseek_fasta} specified but not found.",
            file=logFile,
            print_also=True,
        )
        logFile.close()
        sys.exit(5)

    use_sparse = not all_pairs and num_blocks > 1 and args.sparse_loading

    # Make Predictions
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    # Only create this queue if it is going to be used.
    pair_done_queue = mp.Queue() if (use_sparse or num_blocks > 3) else None

    # This uses the pytorch spawn function to start a bunch of processes using spawn
    # Apparently, spawn (method) is required when using CUDA in the processes

    if device == -1:  # Use all GPUs
        n_gpu = torch.cuda.device_count()
        _ = mp.spawn(
            _predict,
            args=(
                modelPath,
                input_queue,
                output_queue,
                args.store_cmaps,
                use_fs,
                pair_done_queue,
            ),
            nprocs=n_gpu,
            join=False,
        )
    else:  # Use CPU or specific GPU
        p = mp.Process(
            target=_predict,
            args=(
                device,  # "cpu" for CPU, or an index for a GPU
                modelPath,
                input_queue,
                output_queue,
                args.store_cmaps,
                use_fs,
                pair_done_queue,
            ),
        )
        p.start()
        n_gpu = 1

    outPathAll = f"{outPath}.tsv"
    outPathPos = f"{outPath}.positive.tsv"
    if args.store_cmaps:
        cmapPath = f"{outPath}.cmaps.h5"
    else:
        cmapPath = None

    # The writer needs to be seperate from the main process so that writing can start before all pairs are in the queue
    # We are still passing a large array (list of seq names) but it saves us passing names of all pairs of strings
    # Note that we can't share a queue between spawned and forked processes, so this also needs to be spawned
    write_proc = mp.Process(
        target=_writer,
        args=(
            all_prots,
            outPathAll,
            outPathPos,
            cmapPath,
            n_pairs,
            threshold,
            output_queue,
        ),
    )
    write_proc.start()

    # Create the pool we will use to load embeddings
    loadpool = LoadingPool(embPath, n_jobs=args.load_proc)

    block_size = math.ceil(n_prots / num_blocks)

    def get_bounds(block):
        start = block * block_size
        end = min(start + block_size, n_prots)
        return (start, end)  # all_prots[start:end]

    # flag indicates the number assigned to a previous pairs of blocks to wait for
    # before loading the specified block of or indices of proteins
    def load_prots(block, flag=None, indices=None):
        if flag:  # Should be a positive int if not None
            last = pair_done_queue.get()
            # assert last == flag #Since the blocks are done in the provided order, this is just a sanity check for the single GPU case
            # ^ above may fail in multi-GPU mode, so switch to this loop...
            # Now, will naively check for out of order finishes and/or wait for the expected block to finish
            # But, could cause the GPUs to idle for a bit if all other block(s) finish before the expected one
            while last != flag:
                log(
                    f"Found that numbered pair {last} was completed while expecting pair {flag}",
                    file=logFile,
                    print_also=False,
                )
                new_last = pair_done_queue.get()  # By getting before re-putting, we avoid constantly looping while only the unexpected block has finished
                pair_done_queue.put(last)
                last = new_last
            log(
                f"Finished numbered pair {flag} before loading data for block {block}",
                file=logFile,
                print_also=False,
            )
        if indices is not None:
            prot_list = [all_prots[i] for i in indices]
        else:
            prot_list = all_prots[slice(*get_bounds(block))]
        embeds = loadpool.load(prot_list)
        if use_fs:
            fsDict = parse_from_list(foldseek_fasta, prot_list)
            fsOneHotList = [
                get_foldseek_onehot(n0, p0.shape[1], fsDict, fold_vocab)
                .unsqueeze(0)
                .share_memory_()
                for n0, p0 in zip(prot_list, embeds)
            ]
            return (embeds, fsOneHotList)
        return embeds

    # FANCY LOADING
    # We move through pairs of blocks in the order (0,0), (0,1), ..., (0,n), (1,n), ..., (1,3), (1,1), (1,2), (2,2), ...
    # This requires loading at most one new block into memory for each additional pair of proteins
    if not use_sparse:

        def submit_self_block(block, embeddings):
            start, end = get_bounds(block)
            for i0 in range(start, end):
                if use_fs:
                    p0 = embeddings[0][i0 - start]
                    fs0 = embeddings[1][i0 - start]
                else:
                    p0 = embeddings[i0 - start]
                for i1 in range(i0 + 1, end):
                    if all_pairs or pairs_bool[i0, i1]:
                        if use_fs:
                            p1 = embeddings[0][i1 - start]
                            fs1 = embeddings[1][i1 - start]
                            tup = (i0, i1, p0, p1, fs0, fs1)
                        else:
                            p1 = embeddings[i1 - start]
                            tup = (i0, i1, p0, p1)
                        input_queue.put(tup)
            log(f"Self-block submitted: {block}", file=logFile, print_also=False)

        def submit_block(block1, block2, embeddings1, embeddings2, flag=None):
            start1, end1 = get_bounds(block1)
            start2, end2 = get_bounds(block2)
            for i0 in range(start1, end1):
                if use_fs:
                    p0 = embeddings1[0][i0 - start1]
                    fs0 = embeddings1[1][i0 - start1]
                else:
                    p0 = embeddings1[i0 - start1]
                for i1 in range(start2, end2):
                    if all_pairs or pairs_bool[i0, i1]:
                        if use_fs:
                            p1 = embeddings2[0][i1 - start2]
                            fs1 = embeddings2[1][i1 - start2]
                            tup = (i0, i1, p0, p1, fs0, fs1)
                        else:
                            p1 = embeddings2[i1 - start2]
                            tup = (i0, i1, p0, p1)
                        input_queue.put(tup)
            if (
                flag and pair_done_queue
            ):  # Should be a positive int if not None, only do if PDQ is in use
                input_queue.put((None, flag))
            log(
                f"Block submitted: {block1}, {block2} with blocking number {flag}",
                file=logFile,
                print_also=False,
            )

        data1 = load_prots(0)
        data2 = None
        data3 = None
        cur_waiting_pair = 0
        for i in range(num_blocks):
            if i % 2 == 0:
                # Do self block
                submit_self_block(i, data1)
                if i == 0 and num_blocks > 1:
                    data3 = load_prots(1)
                for j in range(i + 1, num_blocks):  # Move up other blocks
                    if j == i + 1:
                        data2 = data3  # Reuse previously loaded data
                        if j != num_blocks - 2:
                            data3 = None  # Remove this reference, except when j is the penultimate block here
                    else:
                        # The first time this is called (i=0, j=2), this will wait for flag=0, which will skip (0 is False), as we don't need to wait
                        data2 = load_prots(j, flag=cur_waiting_pair - 1)
                    cur_waiting_pair += 1
                    submit_block(i, j, data1, data2, flag=cur_waiting_pair)
            else:
                if (
                    i == num_blocks - 1
                ):  # Reuse data in the last loop (condition depends on parity of num_blocks)
                    data1 = data2
                elif i == num_blocks - 2:
                    data1 = data3
                else:
                    data1 = load_prots(i, flag=cur_waiting_pair - 1)
                for j in range(num_blocks - 1, i + 2, -1):  # Move down other blocks
                    cur_waiting_pair += 1
                    submit_block(i, j, data1, data2, flag=cur_waiting_pair)
                    data2 = load_prots(j - 1, flag=cur_waiting_pair - 1)
                if i < num_blocks - 2:
                    # This block won't be blocked on, as we will submit two more non-self blocks before blocking on the first of those
                    submit_block(i, i + 2, data1, data2, flag=None)
                    # But, we now want to block on the immediately previous non-self block for the next load
                    # So, we still incrment cwp, even though it means that a flag is skipped
                    cur_waiting_pair += 1
                    data3 = data2
                # do self-block
                submit_self_block(i, data1)
                # We do the last non-self pair after as the other block (j=i+1) will become the next i
                if i < num_blocks - 1:
                    if i != num_blocks - 2:
                        data2 = load_prots(i + 1, flag=cur_waiting_pair - 1)
                    cur_waiting_pair += 1
                    submit_block(i, i + 1, data1, data2, flag=cur_waiting_pair)
                data1 = data2

    # NON-FANCY LOADING
    # Go through pairs of blocks in order (0,0), (0,1), ..., (0,n), (0, 1), (1,1), (1,2), ...
    # For each pair of blocks, load only the required proteins into memory
    else:
        # Takes two lists of protein indices and corresponding embeddings
        # Checks each pair in pairs_bool and submits if specified.
        # Done like this (versus passing a list of pairs) so embeddings can be accessed by protein index
        def submit_pairs(prots1, prots2, embeddings1, embeddings2, flag):
            for j0, i0 in enumerate(prots1):
                if use_fs:
                    p0 = embeddings1[0][j0]
                    fs0 = embeddings1[1][j0]
                else:
                    p0 = embeddings1[j0]
                row = pairs_bool[i0]
                for j1, i1 in enumerate(prots2):
                    if row[i1]:
                        if use_fs:
                            p1 = embeddings2[0][j1]
                            fs1 = embeddings2[1][j1]
                            tup = (i0, i1, p0, p1, fs0, fs1)
                        else:
                            p1 = embeddings2[j1]
                            tup = (i0, i1, p0, p1)
                        input_queue.put(tup)
            input_queue.put((None, flag))

        # Iterate through all pairs of blocks, identifying the proteins that are part of all pairs of interest
        # between those blocks. No data resue as different subsets of proteins may be needed each time a block
        # is considered. We use blocking for all submisssions to cap the maximum embedding memory usage.
        cur_waiting_pair = 0
        for i in range(0, num_blocks):
            start1, end1 = get_bounds(i)
            # self-pair
            block_pairs = pairs_bool[start1:end1, start1:end1]
            prots_needed1 = (
                np.nonzero(block_pairs.any(axis=0) | block_pairs.any(axis=1))[0] + start1
            )
            if len(prots_needed1) > 0:
                data1 = load_prots(
                    i, flag=max(cur_waiting_pair - 1, 0), indices=prots_needed1
                )
                cur_waiting_pair += 1
                submit_pairs(prots_needed1, prots_needed1, data1, data1, cur_waiting_pair)
                log(
                    f"Self-block submitted: {i} with blocking number {cur_waiting_pair}",
                    file=logFile,
                    print_also=False,
                )
                data1 = None

            for j in range(i + 1, num_blocks):  # other blocks
                start2, end2 = get_bounds(j)
                block_pairs = pairs_bool[start1:end1, start2:end2]
                prots_needed1 = np.nonzero(block_pairs.any(axis=1))[0] + start1
                prots_needed2 = np.nonzero(block_pairs.any(axis=0))[0] + start2
                if len(prots_needed1) > 0:
                    data1 = load_prots(
                        i, flag=max(cur_waiting_pair - 1, 0), indices=prots_needed1
                    )
                    data2 = load_prots(j, flag=None, indices=prots_needed2)
                    cur_waiting_pair += 1
                    submit_pairs(
                        prots_needed1, prots_needed2, data1, data2, cur_waiting_pair
                    )
                    log(
                        f"Block submitted: {i}, {j} with blocking number {cur_waiting_pair}",
                        file=logFile,
                        print_also=False,
                    )
                    data1 = None
                    data2 = None

    loadpool.shutdown()
    # Signal workers to stop after processing all tasks
    for _ in range(n_gpu):
        input_queue.put(None)

    # The writing process knows how may pairs to expect, so we only have to wait for it to finish.
    write_proc.join()

    log("All predictions completed", file=logFile, print_also=True)
    logFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
