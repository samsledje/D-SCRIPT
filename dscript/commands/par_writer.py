from __future__ import annotations

import h5py
import numpy as np
from tqdm import tqdm


# Writer process function for parallel prediction
def _writer(
    all_prots, outPathAll, outPathPos, cmapPath, n_pairs, threshold, output_queue
):
    n = 0
    f = open(outPathAll, "w+")
    pos_f = open(outPathPos, "w+")
    store_cmaps = cmapPath is not None
    if store_cmaps:
        cmap_file = h5py.File(cmapPath, "w")
    with tqdm(total=n_pairs, desc="Writing Predictions") as pbar:
        while n < n_pairs:
            res = output_queue.get()
            n += 1
            i0, i1, p = res[:3]
            n0 = all_prots[i0]
            n1 = all_prots[i1]
            f.write(f"{n0}\t{n1}\t{p}\n")
            if p >= threshold:
                pos_f.write(f"{n0}\t{n1}\t{p}\n")
                if store_cmaps:
                    cm = res[3]
                    cm_np = cm.numpy()
                    dset = cmap_file.require_dataset(
                        f"{n0}x{n1}", cm_np.shape, np.float32
                    )
                    dset[:] = cm_np
            pbar.update(1)

    f.close()
    pos_f.close()
    if store_cmaps:
        cmap_file.close()
