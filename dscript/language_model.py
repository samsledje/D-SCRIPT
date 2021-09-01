import logging as logg
import sys

import h5py
import torch
from Bio import SeqIO
from tqdm import tqdm

from .alphabets import Uniprot21
from .pretrained import get_pretrained


def lm_embed(sequence, use_cuda=False, verbose=True):
    """
    Embed a single sequence using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param sequence: Input sequence to be embedded
    :type sequence: str
    :param use_cuda: Whether to generate embeddings using GPU device [default: False]
    :type use_cuda: bool
    :return: Embedded sequence
    :rtype: torch.Tensor
    """

    model = get_pretrained("lm_v1", verbose=verbose)
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()
    model.eval()

    with torch.no_grad():
        alphabet = Uniprot21()
        es = torch.from_numpy(alphabet.encode(sequence.encode("utf-8")))
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
            logg.info(
                f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}"
            )
    else:
        if verbose:
            logg.info("Using CPU")

    if verbose:
        logg.info("Loading Model...")

    model = get_pretrained("lm_v1")
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()

    model.eval()
    if verbose:
        logg.info("Loading Sequences...")

    seq_records = list(SeqIO.parse(fastaPath, "fasta"))

    alphabet = Uniprot21()
    encoded_seqs = {}
    for rec in tqdm(seq_records):
        es = torch.from_numpy(alphabet.encode(rec.seq.encode("utf-8")))
        if use_cuda:
            es = es.cuda()
        encoded_seqs[rec.name] = es
    if verbose:
        num_seqs = len(encoded_seqs)
        logg.info("{} Sequences Loaded".format(num_seqs))
        logg.info(
            "Approximate Storage Required (varies by average sequence length): ~{}GB".format(
                num_seqs * (1 / 125)
            )
        )

    logg.info("Storing to {}...".format(outputPath))
    with h5py.File(outputPath, "w") as h5fi, torch.no_grad():
        try:
            for req in tqdm(seq_records, total=len(seq_records)):
                if req.name not in h5fi:
                    enc = alphabet.encode(req.seq.encode("utf-8"))
                    x = torch.from_numpy(enc).long().unsqueeze(0)
                    z = model.transform(x)
                    h5fi.create_dataset(
                        req.name, data=z.cpu().numpy(), compression="lzf"
                    )
        except KeyboardInterrupt:
            h5fi.close()
            sys.exit(1)
