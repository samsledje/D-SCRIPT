import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import lru_cache
import sys
from dscript.utils import log


def bucket_prefix2(acc: str) -> str:
    """Two-level prefix partition to keep directories balanced but shallow."""
    return f"{acc[:1]}/{acc[:2]}"


def add_batch_dim_if_needed(x: torch.Tensor) -> torch.Tensor:
    # Old HDF5 tensors were [1, L, D]; new .pt tensors are [L, D].
    # This makes [L, D] -> [1, L, D], and leaves [B, L, D] unchanged.
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x


class EmbeddingLoader:
    def __init__(self, embedding_dir_name, protein_names, num_workers=4):
        self.embedding_dir = Path(embedding_dir_name)
        self.embeddings_cpu = {}
        self.missing = []

        self._load_to_cpu_parallel(protein_names, num_workers)

        if self.missing:

            log(
                f"Missing files for {len(self.missing)} proteins (e.g., {self.missing[:8]})"
            )
            sys.exit(1)

        log(f"Loaded {len(self.embeddings_cpu)} embeddings to CPU.")

    def _load_one(self, prot):
        """Load a single .pt file to CPU."""
        subfolder = bucket_prefix2(prot)
        fp = self.embedding_dir / subfolder / f"{prot}.pt"
        # print(fp)
        if not fp.exists():
            return prot, None
        try:
            emb = torch.load(fp, map_location="cpu", weights_only=True)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu()
            return prot, emb
        except Exception as e:
            log(f"Error loading {prot}: {e}")
            return prot, None

    def _load_to_cpu_parallel(self, protein_names, num_workers):
        """Load all .pt files to CPU in parallel."""
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self._load_one, sorted(protein_names)),
                    total=len(protein_names),
                    desc="Loading .pt embeddings",
                )
            )

        for prot, emb in results:
            if emb is not None:
                self.embeddings_cpu[prot] = emb
            else:
                self.missing.append(prot)

    def __getitem__(self, prot_name):
        if prot_name not in self.embeddings_cpu:
            raise KeyError(f"Protein {prot_name} not found")
        return self.embeddings_cpu[prot_name]
