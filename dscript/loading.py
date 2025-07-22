import queue
import sys

import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from .load_worker import _hdf5_load_partial_func


# Seperate managing the pool from the loading function
# to allow the calling process to keep the pool around
class LoadingPool:
    def __init__(self, file_path, n_jobs=-1, timeout=60):
        if n_jobs < 1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs
        # Remark: the data loading itself is multi-threaded under the hood
        # So, CPU utilization is high regardless of how many processes are used
        # But throughput is a bit faster using multiple processes for some reason

        # Note: Using spawn (or torch.mp.spawn) caused errors, make sure to use fork
        ctx = mp.get_context("fork")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.queue_timeout = timeout

        self.pool = ctx.Pool(
            processes=self.n_jobs,
            initializer=_hdf5_load_partial_func,
            initargs=(self.input_queue, self.output_queue, file_path),
        )
        self.pool.close()

    # Will always return a list in the order that inputs are received
    def load(self, keys, progress=False):
        count = 0
        for key in keys:
            self.input_queue.put((key, count))
            count += 1
        embeddings = [None] * len(keys)
        loaded = 0
        if progress:
            pbar = tqdm(total=count, desc="Loading Embeddings")
        while loaded < count:
            try:
                res = self.output_queue.get(timeout=self.queue_timeout)
                i, emb = res
                embeddings[i] = emb
                if progress:
                    pbar.update(1)
                loaded += 1
            except queue.Empty:
                logger.error("Loading embeddings timed out -- see errors above.")
                sys.exit(7)
        if progress:
            pbar.close()
        return embeddings

    # Basically does load and shutdown together - based on older version
    def load_once(self, keys, progress=True):
        count = 0
        for key in keys:
            self.input_queue.put((key, count))
            count += 1
        embeddings = [None] * len(keys)
        done_count = 0
        if progress:
            pbar = tqdm(total=len(keys), desc="Loading Embeddings")
        for _ in range(self.n_jobs):
            self.input_queue.put(None)
        while done_count < self.n_jobs:
            res = self.output_queue.get()
            if res is None:  # This makes really sure that each job is finished processing
                done_count += 1
            else:
                i, emb = res
                embeddings[i] = emb
                if progress:
                    pbar.update(1)
        if progress:
            pbar.close()
        self.pool.join()
        return embeddings

    def shutdown(self):
        for _ in range(self.n_jobs):
            self.input_queue.put(None)
        for _ in range(self.n_jobs):
            _ = self.output_queue.get(timeout=self.queue_timeout)
        self.pool.join()
