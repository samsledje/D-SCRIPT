import torch
import pandas as pd
import numpy as np
from collections import Counter

from pathlib import Path
from loguru import logger
from .utils import log

def parse_parquet_embeddings(n0, size_n0, patch_dir, patch_dim = 384):
    patch_path = Path(patch_dir) / f"{n0}.parquet"

    if not patch_path.exists():
        logger.debug(f"{n0}: no parquet found, returning zeros [{size_n0}, {patch_dim}]")
        return torch.zeros(size_n0, patch_dim, dtype=torch.float32)

    df = pd.read_parquet(patch_path, columns=["residue_seq_pos", "patch_embedding"])

    max_seq_pos = df["residue_seq_pos"].max()
    if max_seq_pos >= size_n0:
        logger.warning(
            f"{n0}: max residue_seq_pos ({max_seq_pos}) >= size_n0 ({size_n0}). "
            f"Embedding may be truncated or size_n0 is incorrect."
        )
        return torch.zeros(size_n0, patch_dim, dtype=torch.float32)

    output = torch.zeros(size_n0, patch_dim, dtype=torch.float32)

    for _, row in df.iterrows():
        seq_pos = row["residue_seq_pos"]
        if seq_pos is None or seq_pos >= size_n0:
            continue
        emb = torch.tensor(row["patch_embedding"], dtype=torch.float32)
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            continue
        output[seq_pos] = emb

    return output

def parse_parquet_codes(proteins, patch_dir, ncodes = 10, max_prevalence = 40.0):
    patch_dir = Path(patch_dir)
    out_dim   = ncodes * 3
    parquet_files = sorted(patch_dir.glob("*.parquet"))
    n_proteins  = len(parquet_files)

    # compute excluded codes internally
    logger.info(f"Computing code prevalence across {n_proteins} proteins")
    code_protein_counter = Counter()
    for f in parquet_files:
        df = pd.read_parquet(f, columns=["codes"])
        protein_codes = set()
        for codes in df["codes"]:
            codes = np.stack(codes)
            for c in codes:
                protein_codes.add(f"{c[0]}-{c[1]}-{c[2]}")
        code_protein_counter.update(protein_codes)

    excluded_codes = {
        code for code, count in code_protein_counter.items()
        if count / n_proteins > max_prevalence
    }
    logger.info(f"Excluding {len(excluded_codes)} codes with prevalence >{max_prevalence}%")

    # parse embeddings
    patch_codes = {}

    filtered_out = 0
    other_count = 0

    for prot_name, seq_emb in proteins.items():
        size_n0    = seq_emb.shape[0]
        patch_path = patch_dir / f"{prot_name}.parquet"

        output = torch.zeros(size_n0, out_dim, dtype=torch.float32)

        if not patch_path.exists():
            patch_codes[prot_name] = output
            continue

        df = pd.read_parquet(patch_path, columns=["residue_seq_pos", "codes"])
        df = df.dropna(subset=["residue_seq_pos"])
        df["residue_seq_pos"] = df["residue_seq_pos"].astype(int)

        max_seq_pos = df["residue_seq_pos"].max()
        if max_seq_pos >= size_n0:
            print(f"investigate {prot_name}")
            patch_codes[prot_name] = output
            continue

        for _, row in df.iterrows():
            seq_pos = row["residue_seq_pos"]
            codes   = np.stack(row["codes"])

            filtered = [
                c for c in codes
                if f"{c[0]}-{c[1]}-{c[2]}" not in excluded_codes
            ]

            if len(filtered) < ncodes:
                filtered_out = filtered_out + 1
                continue
            else:
                other_count = other_count + 1

            top_codes = np.stack(filtered[:ncodes])  # [ncodes, 3]
            output[seq_pos] = torch.from_numpy(top_codes.flatten().astype(np.float32))

        patch_codes[prot_name] = output

    return patch_codes
