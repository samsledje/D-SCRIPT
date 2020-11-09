import sys
from src.embed import embed_from_fasta

parser = argparse.ArgumentParser("Script for training protein interaction prediction model")
parser.add_argument("in_path", help="Sequences to be embedded (.fasta)", required=True)
parser.add_argument("out_path", help="File to write embeddings (.h5)", required=True)
parser.add_argument("-d", "--device", default=-1, help="Compute device to use")

arsg = parser.parse_args
in_path = args.in_path
out_path = args.out_path
device = args.device

embed_from_fasta(in_path, out_path, device)
