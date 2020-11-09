"""
Generate new embeddings using pre-trained language model
"""

import argparse
from dscript.lm_embed import embed_from_fasta


def add_args(parser):
    parser.add_argument("--seqs", help="Sequences to be embedded", required=True)
    parser.add_argument("--outfile", help="h5 file to write results", required=True)
    parser.add_argument("-d", "--device", default=-1, help="Compute device to use")
    return parser


def main(args):
    inPath = args.fasta
    outPath = args.outfile
    device = args.device
    embed_from_fasta(inPath, outPath, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
