import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dscript.utils import (
    RBF,
    augment_data,
    get_local_or_download,
    gpu_mem,
    plot_PR_curve,
    plot_ROC_curve,
)


def test_get_local_or_download():
    destination_path = Path("scratch/ecoli.fasta")
    source_path = (
        "https://github.com/samsledje/D-SCRIPT/raw/dev/data/seqs/ecoli.fasta"
    )

    if destination_path.exists():
        os.remove(destination_path.resolve())
    assert not destination_path.exists()

    pth = get_local_or_download(
        destination=str(destination_path), source=source_path
    )
    assert destination_path.exists()
    assert Path(pth) == destination_path.resolve()

    pth_local = get_local_or_download(
        destination=str(destination_path), source=source_path
    )
    assert Path(pth_local).resolve() == destination_path.resolve()


def test_gpu_mem():
    if torch.cuda.is_available():
        in_use, total = gpu_mem(0)
        assert in_use >= 0
        assert total > 0
    else:
        return True


def test_augment_data():
    df = pd.DataFrame([["a", "b", 0], ["c", "d", 0], ["e", "f", 1]])
    aug_df = augment_data(df)
    aug_test = pd.DataFrame(
        [
            ["a", "b", 0],
            ["c", "d", 0],
            ["e", "f", 1],
            ["b", "a", 0],
            ["d", "c", 0],
            ["f", "e", 1],
        ]
    )
    pd.testing.assert_frame_equal(aug_test, aug_df)


def test_plotting():
    y = torch.tensor([0, 1, 1, 0, 0, 1])
    phat = torch.tensor([0.1, 0.2, 0.4, 0.3, 0.5, 0.6])
    plot_ROC_curve(y, phat, show=False)
    plot_PR_curve(y, phat, show=False)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_rbf():
    assert np.isnan(RBF(0.0))
    assert np.allclose(RBF(0.5), 0.77880)
    assert np.allclose(RBF(1.0), 0.60653)
