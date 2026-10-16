"""
Frozen copy of the training command from the upstream D-SCRIPT repository:

    https://github.com/samsledje/D-SCRIPT, branch tt3d_new,
    commit 36577435c882e9699c8e3ab7509d3e887d208fae

``dscript train`` runs this code. It is the "upstream" arm of an A/B/C
comparison with ``train_serial`` and ``train_batched``, so it must not change.

Every file is byte-for-byte the upstream file, except import lines in
train.py and parallel_embedding_loader.py, which point at the copies in this
folder instead of the package root. The copied modules (models/, foldseek, utils, glider, fasta,
parallel_embedding_loader, loading, load_worker) are the upstream versions,
not the ones elsewhere in this package, so edits there cannot change it.
Nothing outside this folder imports it except the ``train`` command.
"""
