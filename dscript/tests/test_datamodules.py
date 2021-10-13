import torch

from dscript.datamodules import CachedFasta, CachedH5, collate_pairs_fn


def test_collate_pairs_fn():
    args = (
        [torch.tensor([0]), torch.tensor([3, 4]), torch.tensor([0])],
        [torch.tensor([1]), torch.tensor([5, 6]), torch.tensor([0])],
        [torch.tensor([2]), torch.tensor([7, 8]), torch.tensor([0])],
    )
    x0_col, x1_col, y_col = collate_pairs_fn(args)
    # assert x0_col.shape == (3, 1)
    # assert (x0_col <= 2).all()
    # assert x1_col.shape == (3, 2)
    # assert (x1_col > 2).all()
    # assert y_col.shape == (3, 1)
    # assert (y_col == 0).all()
    assert len(x0_col) == 3
    assert len(x1_col) == 3
    assert y_col.shape == (3, 1)
    assert (y_col == 0).all()
