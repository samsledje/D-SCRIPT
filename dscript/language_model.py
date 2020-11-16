import os, sys
import subprocess as sp
import random
import torch
import h5py
from .fasta import parse, parse_directory, write
from .alphabets import Uniprot21
from .models.embedding import SkipLSTM
from datetime import datetime

EMBEDDING_STATE_DICT = "/afs/csail/u/s/samsl/db/embedding_state_dict.pt"


def embed_from_fasta(fastaPath, outputPath, device=0, verbose=False):
    """
    Embed sequences using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_

    :param fastaPath: Input sequence file (``.fasta`` format)
    :type fastaPath: str
    :param outputPath: Output embedding file (``.h5`` format)
    :type outputPath: str
    :param device: Compute device to use for embeddings [default: 0]
    :type device: int
    :param verbose: Print embedding progress
    :type verbose: bool
    """
    use_cuda = (device != -1) and torch.cuda.is_available()
    if device >= 0:
        torch.cuda.set_device(device)
        if verbose:
            print(f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}")
    else:
        if verbose:
            print("# Using CPU")

    if verbose:
        print("# Loading Model...")
    model = SkipLSTM(21, 100, 1024, 3)
    model.load_state_dict(torch.load(EMBEDDING_STATE_DICT))
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()

    if verbose:
        print("# Loading Sequences...")
    names, seqs = parse(open(fastaPath, "rb"))
    alphabet = Uniprot21()
    encoded_seqs = [torch.from_numpy(alphabet.encode(s)) for s in seqs]
    if use_cuda:
        encoded_seqs = [x.cuda() for x in encoded_seqs]
    if verbose:
        print("# {} Sequences Loaded".format(len(encoded_seqs)))

    h5fi = h5py.File(outputPath, "w")

    print("# Storing to {}...".format(outputPath))
    with torch.no_grad():
        for i, (n, x) in enumerate(zip(names, encoded_seqs)):
            x = x.long().unsqueeze(0)
            z = model.transform(x)
            name = n.decode("utf-8")
            h5fi.create_dataset(name, data=z.cpu().numpy(), compression="lzf")
            if verbose and i % 100 == 0:
                print("# {} sequences processed...".format(i), file=sys.stderr)

    h5fi.close()


def embed_from_directory(directory, outputPath, device=0, verbose=False, extension=".seq"):
    """
    Embed all files in a directory in ``.fasta`` format using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_

    :param directory: Input directory (``.fasta`` format)
    :type directory: str
    :param outputPath: Output embedding file (``.h5`` format)
    :type outputPath: str
    :param device: Compute device to use for embeddings [default: 0]
    :type device: int
    :param verbose: Print embedding progress
    :type verbose: bool
    :param extension: Extension of all files to read in
    :type extension: str
    """
    nam, seq = parse_directory(directory, extension=extension)
    fastaPath = f"{directory}/allSeqs.fa"
    if os.path.exists(fastaPath):
        fastaPath = f"{fastaPath}.{int(datetime.utcnow().timestamp())}"
    write(nam, seq, open(fastaPath, "w"))
    embed_from_fasta(fastaPath, outputPath, device, verbose)
