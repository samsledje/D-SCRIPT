import os
import urllib
import shutil
import atexit
import h5py
import pandas as pd
from typing import Optional, List
from pathlib import Path
from functools import lru_cache

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from . import __version__
from .fasta import parse
from .utils import get_local_or_download
from .language_model import embed_from_fasta


def collate_pairs_fn(args):
    """
    Collate function for PyTorch data loader.
    """
    x0 = pad_sequence([a[0] for a in args], batch_first=True)
    x1 = pad_sequence([a[1] for a in args], batch_first=True)
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)


class CachedH5:
    def __init__(self, filePath: str):
        self.filePath = filePath
        self.h5file = h5py.File(self.filePath, "r")
        atexit.register(self.cleanup)

    def cleanup(self):
        self.h5file.close()

    @lru_cache(maxsize=1000)
    def __getitem__(self, x):
        return torch.from_numpy(self.h5file[x][:]).squeeze()


class PairedEmbeddingDataset(Dataset):
    """
    Dataset to be used by the PyTorch data loader for pairs of sequences and their labels.

    :param x0: List of first name in the pair
    :param x1: List of second name in the pair
    :param y: List of labels
    :param embedding: Embeddings
    """

    def __init__(self, pair_df: pd.DataFrame, embedding: CachedH5):
        self.x0 = pair_df[0]
        self.x1 = pair_df[1]
        self.y = pair_df[2]
        self.embedding = embedding

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, i):
        return (
            self.embedding[self.x0.iloc[i]],
            self.embedding[self.x1.iloc[i]],
            torch.tensor(self.y.iloc[i]),
        )


class PPIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sequence_path: str,
        pair_path: str,
        data_dir: str = os.getcwd(),
        batch_size: int = 64,
        shuffle: bool = True,
        train_val_split: List[float] = [0.9, 0.1],
    ):
        super(PPIDataModule).__init__()

        self.data_dir = Path(data_dir)
        self.sequence_path = Path(sequence_path)
        self.pair_path = Path(pair_path)

        # If sequence_path is a URL, prepare for download
        url_seq = urllib.parse.urlparse(sequence_path)
        if url_seq.scheme in ["http", "https"]:
            self.sequence_path = self.data_dir / Path(
                os.path.basename(url_seq.path)
            )
            self.sequence_url = sequence_path
        else:
            self.sequence_url = None

        # If pair_path is a URL, prepare for download
        url_pair = urllib.parse.urlparse(pair_path)
        if url_pair.scheme in ["http", "https"]:
            self.pair_path = self.data_dir / Path(
                os.path.basename(url_pair.path)
            )
            self.pair_url = pair_path
        else:
            self.pair_url = None

        # Hyperparams
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_val_split = train_val_split

    def prepare_data(self):
        """
        Prepare data for DataModule.
        """
        get_local_or_download(self.sequence_path, self.sequence_url)

        # If sequence_path is not embeddings, embed
        if not self.sequence_path.suffix == ".h5":
            assert not self.sequence_path.with_suffix(".h5").exists()
            embed_from_fasta(
                self.sequence_path,
                self.sequence_path.with_suffix(".h5"),
                verbose=True,
            )
            self.sequence_path = self.sequence_path.with_suffix(".h5")

    def setup(self, stage: Optional[str] = None):

        self.embeddings = CachedH5(self.sequence_path)

        self.full_df = pd.read_csv(self.pair_path, sep="\t", header=None)
        self.full_df = self.full_df.sort_values([0, 1])

        self.train_df, self.val_df = train_test_split(
            self.full_df,
            train_size=self.train_val_split[0],
            test_size=self.train_val_split[1],
        )

        self.data_train = PairedEmbeddingDataset(
            self.train_df, self.embeddings
        )

        self.data_val = PairedEmbeddingDataset(self.val_df, self.embeddings)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_pairs_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_pairs_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_pairs_fn,
        )

    def teardown(self):
        self.embeddings.cleanup()
