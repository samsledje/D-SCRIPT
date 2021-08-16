import dscript


def test_log():
    dscript.utils.log("Testing logging")


def test_paired_dataset():
    pds = dscript.utils.PairedDataset([1, 2, 3], [4, 5, 6], [1, 0, 1])
    assert len(pds) == 3
    assert pds[0] == (1, 4, 1)
