import os, sys
import subprocess as sp
import random
import torch
import h5py
from tqdm import tqdm
from .fasta import parse, parse_directory, write
from .alphabets import Uniprot21
from .models.embedding import SkipLSTM
from datetime import datetime


def get_state_dict(version=1):
    """
    Download the pre-trained language model if not already exists on local device. This is required because the model state dict is too large to be stored on Github.

    :param version: Version of language model to download [default: 1]
    :type version: int
    :return: Path to state dictionary for pre-trained language model
    :rtype: str
    """
    state_dict_basename = f"lm_model_v{version}.pt"
    state_dict_basedir = os.path.dirname(os.path.realpath(__file__))
    state_dict_fullname = f"{state_dict_basedir}/{state_dict_basename}"
    state_dict_url = f"http://cb.csail.mit.edu/cb/dscript/data/{state_dict_basename}"
    if not os.path.exists(state_dict_fullname):
        try:
            import urllib.request
            import shutil
            print("Downloading Language Model from {}...".format(state_dict_url))
            with urllib.request.urlopen(state_dict_url) as response, open(state_dict_fullname, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            print("Unable to download language model - {}".format(e))
            sys.exit(1)
    return state_dict_fullname


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
    model = SkipLSTM(21, 100, 1024, 3)
    lm_state_dict = get_state_dict()
    if verbose:
        print('# Using model from {}'.format(lm_state_dict))
    model.load_state_dict(torch.load(lm_state_dict))
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()

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
