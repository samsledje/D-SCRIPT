import h5py
import torch

#Worker process function for loading embeddings
def _hdf5_load_partial_func(qin, qout, file_path):
    """
    Helper function for load_hdf5_parallel
    """
    with h5py.File(file_path, "r") as fi:
        for k in iter(qin.get, None):
            emb = torch.from_numpy(fi[k][:])
            emb.share_memory_()
            qout.put((k, emb))
        qout.put(None)
    
