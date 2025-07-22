"""
Make new predictions between two protein sets using blocked, multi-GPU pariwise inference  with a pre-trained model.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
from collections.abc import Callable
from typing import NamedTuple

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


class BipartitePredictionArguments(NamedTuple):
    cmd: str
    protA: str
    protB: str
    model: str | None
    embedA: str
    embedA: str | None
    foldseekA: str | None
    foldseekB: str | None
    outfile: str | None
    device: str | None
    thresh: float | None
    load_proc: int | None
    blocksA: int | None
    blocksB: int | None
    func: Callable[[BipartitePredictionArguments], None]


def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """
    parser.add_argument(
        "--protA",
        required=True,
        help="A text file with protein IDs, one on each line. All pairs between proteins in this file and proteins in protB will be predicted",
    )
    parser.add_argument(
        "--protB",
        required=True,
        help="A text file with protein IDs, one on each line. All pairs between proteins in protA and proteins in this file will be predicted",
    )
    parser.add_argument(
        "--model",
        help="Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_human_v1]",
        default="samsl/topsy_turvy_human_v1",
    )
    parser.add_argument(
        "--embedA",
        required=True,
        help="""h5 file with (a superset of) pre-embedded sequences from the file protA. Generate with dscript embed.
        If a single file contains embeddings for both protA and protB, specify it as embedA.""",
    )
    parser.add_argument(
        "--embedB",
        required=False,
        help="h5 file with (a superset of) pre-embedded sequences from the file protB. Generate with dscript embed.",
    )
    parser.add_argument(
        "--foldseekA",
        help="""3di sequences in .fasta format for proteins in protA. Can be generated using `dscript extract-3di.
        Default is None. If provided, TT3D will be run, otherwise default D-SCRIPT/TT will be run.
        If a single file contains 3di sequences for both protA and protB, specify it as foldseekA.
        """,
        default=None,
    )
    parser.add_argument(
        "--foldseekB",
        help="""3di sequences in .fasta format for proteins in protA. Can be generated using `dscript extract-3di.
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
        "--blocksA",
        type=int,
        default=1,
        help="Number of equal-sized blocks to split proteins in protA into. If one set is smuch smaller, it is recommended to set the corresponding # of blocks to 1. Default 1.",
    )
    parser.add_argument(
        "--blocksB",
        type=int,
        default=1,
        help="Number of equal-sized blocks to split proteins in protB into. Default 1.",
    )

    return parser


class ProteinSet:
    def __init__(self, protPath="", blocks=1, logFile=None):
        self.num_blocks = blocks
        try:
            log(f"Loading protein IDs from {protPath}", file=logFile, print_also=True)
            with open(protPath) as f:
                self.all_prots = [
                    line.strip() for line in f if line and not line.isspace()
                ]
        except FileNotFoundError:
            log(f"Proteins file {protPath} not found", file=logFile, print_also=True)
            logFile.close()
            sys.exit(4)
        self.n_prots = len(self.all_prots)
        self.block_size = math.ceil(self.n_prots / self.num_blocks)
        self.loadpool = None
        self.logFile = logFile
        self.use_fs = False

    def set_embed_path(self, path):
        if not os.path.exists(path):
            log(f"Embeddings File {path} not found.", file=self.logFile, print_also=True)
            self.logFile.close()
            sys.exit(3)
        self.embed_path = path

    def set_foldseek_path(self, path):
        if not os.path.exists(path):
            log(
                f"Foldseek FASTA File {path} specified but not found.",
                file=self.logFile,
                print_also=True,
            )
            self.logFile.close()
            sys.exit(5)
        self.foldseek_fasta = path
        self.use_fs = True

    def get_bounds(self, block):
        start = block * self.block_size
        end = min(start + self.block_size, self.n_prots)
        return (start, end)  # all_prots[start:end]

    # flag indicates the number assigned to a previous pairs of blocks to wait for
    # before loading the specified block of proteins
    def load_prots(self, block, flag=None):
        if flag:  # Should be a positive int if not None - 0 is also skipped
            last = self.pair_done_queue.get()  # pdq is set externally
            while last != flag:
                log(
                    f"Found that numbered pair {last} was completed while expecting pair {flag}",
                    file=self.logFile,
                    print_also=False,
                )
                new_last = self.pair_done_queue.get()  # By getting before re-putting, we avoid constantly looping while only the unexpected block has finished
                self.pair_done_queue.put(last)
                last = new_last
            log(
                f"Finished numbered pair {flag} before loading data for block {block}",
                file=self.logFile,
                print_also=False,
            )
        prot_list = self.all_prots[slice(*self.get_bounds(block))]
        embeds = self.loadpool.load(prot_list)  # loadpool is set externally
        if self.use_fs:
            fsDict = parse_from_list(self.foldseek_fasta, prot_list)
            fsOneHotList = [
                get_foldseek_onehot(n0, p0.shape[1], fsDict, fold_vocab)
                .unsqueeze(0)
                .share_memory_()
                for n0, p0 in zip(prot_list, embeds)
            ]
            return (embeds, fsOneHotList)
        return embeds


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

    # With 2 of everything, this will be a lot more readable
    # Initialize and load proteins
    protsA = ProteinSet(args.protA, args.blocksA, logFile)
    protsB = ProteinSet(args.protB, args.blocksB, logFile)
    n_pairs = protsA.n_prots * protsB.n_prots

    # Check embedding files
    protsA.set_embed_path(args.embedA)
    if args.embedB is None:
        shared_embeddings = True
        protsB.set_embed_path(args.embedA)
    else:
        shared_embeddings = False
        protsB.set_embed_path(args.embedB)

    modelPath = args.model
    device = parse_device(args.device, logFile)
    threshold = args.thresh

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

    # Check Foldseek Sequence File(s)
    use_fs = args.foldseekA is not None
    if use_fs:
        protsA.set_foldseek_path(args.foldseekA)
        if args.foldseekB is None:
            protsB.set_foldseek_path(args.foldseekA)
        else:
            protsB.set_foldseek_path(args.foldseekB)

    # Make Predictions
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    # Only create this queue if it is going to be used.
    pair_done_queue = mp.Queue() if (protsA.num_blocks + protsB.num_blocks > 3) else None

    # This uses the pytorch spawn function to start a bunch of processes using spawn
    # Apparently, spawn (method) is required when using CUDA in the processes

    # device = -1 -> use all GPUs
    # See remark in predict_block
    if device == -1:
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
            ),  # Can't pass an open file
            nprocs=n_gpu,
            join=False,
            daemon=True,
        )
    else:
        p = mp.Process(
            target=_predict,
            args=(
                device,
                modelPath,
                input_queue,
                output_queue,
                args.store_cmaps,
                use_fs,
                pair_done_queue,
            ),
        )
        p.daemon = True
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
            protsA.all_prots + protsB.all_prots,
            outPathAll,
            outPathPos,
            cmapPath,
            n_pairs,
            threshold,
            output_queue,
        ),
    )
    write_proc.daemon = True
    write_proc.start()

    # Create the pool we will use to load embeddings
    protsA.loadpool = LoadingPool(protsA.embed_path, n_jobs=args.load_proc)
    protsB.loadpool = (
        protsA.loadpool
        if shared_embeddings
        else LoadingPool(protsB.embed_path, n_jobs=args.load_proc)
    )
    protsA.pair_done_queue = protsB.pair_done_queue = pair_done_queue  # This is shared

    # We move through pairs of blocks in the order (0,0), (0,1), ..., (0,n), (1,n), ..., (1,0), (2,0), (2,1), ...

    def submit_block(block1, block2, embeddings1, embeddings2, flag=None):
        start1, end1 = protsA.get_bounds(block1)
        start2, end2 = protsB.get_bounds(block2)
        for i0 in range(start1, end1):
            if use_fs:
                p0 = embeddings1[0][i0 - start1]
                fs0 = embeddings1[1][i0 - start1]
            else:
                p0 = embeddings1[i0 - start1]
            for i1 in range(start2, end2):
                if use_fs:
                    p1 = embeddings2[0][i1 - start2]
                    fs1 = embeddings2[1][i1 - start2]
                    tup = (i0, i1 + protsA.n_prots, p0, p1, fs0, fs1)
                else:
                    p1 = embeddings2[i1 - start2]
                    tup = (i0, i1 + protsA.n_prots, p0, p1)
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

    cur_waiting_pair = 0
    data2 = protsB.load_prots(0)
    for i in range(protsA.num_blocks):
        data1 = protsA.load_prots(i, flag=max(cur_waiting_pair - 1, 0))
        if i % 2 == 0:
            cur_waiting_pair += 1
            submit_block(i, 0, data1, data2, flag=cur_waiting_pair)  # Reuse data2 = B[0]
            for j in range(1, protsB.num_blocks):
                data2 = protsB.load_prots(j, flag=max(cur_waiting_pair - 1, 0))
                cur_waiting_pair += 1
                submit_block(i, j, data1, data2, flag=cur_waiting_pair)
        else:
            for j in range(protsB.num_blocks - 1, 0, -1):
                cur_waiting_pair += 1
                submit_block(
                    i, j, data1, data2, flag=cur_waiting_pair
                )  # Reuse data2 = B[n]
                data2 = protsB.load_prots(j - 1, flag=max(cur_waiting_pair - 1, 0))
            cur_waiting_pair += 1
            submit_block(i, 0, data1, data2, flag=cur_waiting_pair)

    protsA.loadpool.shutdown()
    if not shared_embeddings:
        protsB.loadpool.shutdown()
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
