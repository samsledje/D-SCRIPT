import os, sys
import subprocess as sp
import random
import torch
import h5py
from tqdm import tqdm
from .fasta import parse, parse_directory, write
from .pretrained import get_pretrained
from .alphabets import Uniprot21
from .models.embedding import SkipLSTM
from datetime import datetime


def lm_embed(sequence, use_cuda=False):
    """
    Embed a single sequence using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param sequence: Input sequence to be embedded
    :type sequence: str
    :param use_cuda: Whether to generate embeddings using GPU device [default: False]
    :type use_cuda: bool
    :return: Embedded sequence
    :rtype: torch.Tensor
    """

    model = get_pretrained("lm_v1")
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()
    model.eval()

    with torch.no_grad():
        alphabet = Uniprot21()
        es = torch.from_numpy(alphabet.encode(sequence.encode('utf-8')))
        x = es.long().unsqueeze(0)
        if use_cuda:
            x = x.cuda()
        z = model.transform(x)
        return z.cpu()


def embed_from_fasta(fastaPath, outputPath, device=0, verbose=False):
    """
    Embed sequences using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param fastaPath: Input sequence file (``.fasta`` format)
    :type fastaPath: str
    :param outputPath: Output embedding file (``.h5`` format)
    :type outputPath: str
    :param device: Compute device to use for embeddings [default: 0]
    :type device: int
    :param verbose: Print embedding progress
    :type verbose: bool
    """
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        if verbose:
            print(f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}")
    else:
        if verbose:
            print("# Using CPU")

    if verbose:
        print("# Loading Model...")
    model = get_pretrained("lm_v1")
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()

    model.eval()
    if verbose:
        print("# Loading Sequences...")
    names, seqs = parse(open(fastaPath, "rb"))
    alphabet = Uniprot21()
    encoded_seqs = []
    for s in tqdm(seqs):
        es = torch.from_numpy(alphabet.encode(s))
        if use_cuda:
            es = es.cuda()
        encoded_seqs.append(es)
    if verbose:
        num_seqs = len(encoded_seqs)
        print("# {} Sequences Loaded".format(num_seqs))
        print("# Approximate Storage Required (varies by average sequence length): ~{}GB".format(num_seqs * (1/125)))

    h5fi = h5py.File(outputPath, "w")

    print("# Storing to {}...".format(outputPath))
    with torch.no_grad():
        try:
            for (n, x) in tqdm(zip(names, encoded_seqs),total=len(names)):
                name = n.decode("utf-8")
                if not name in h5fi:
                    x = x.long().unsqueeze(0)
                    z = model.transform(x)
                    h5fi.create_dataset(name, data=z.cpu().numpy(), compression="lzf")
        except KeyboardInterrupt:
            h5fi.close()
            sys.exit(1)
    h5fi.close()


def embed_from_directory(directory, outputPath, device=0, verbose=False, extension=".seq"):
    """
    Embed all files in a directory in ``.fasta`` format using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

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
