import h5py
import torch
from loguru import logger


# Worker process function for loading embeddings
def _hdf5_load_partial_func(qin, qout, file_path):
    """
    Helper function for load_hdf5_parallel
    """
    # Otherwise, each process would itself try to multithread and exceed the CPU limit
    # Empirically is seems a bit faster to use multiple processes vs one multithreaded,
    # presumably because the processes can also read in parallel.
    # But, I should investigate whether having a small (but >1) number does best
    torch.set_num_threads(1)
    try:
        with h5py.File(file_path, "r") as fi:
            for k, i in iter(qin.get, None):
                emb = torch.from_numpy(fi[k][:])
                emb.share_memory_()
                qout.put((i, emb))
            qout.put(None)
    except Exception as e:
        logger.error(e)
