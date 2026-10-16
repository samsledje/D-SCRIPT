"""
Building blocks shared by every command: settings and process launch, data
reading, the batch type, memory planning, inference, and metrics.

Dependency rule: ``dscript.core`` imports nothing from ``dscript.models``,
``dscript.train`` or ``dscript.commands``. Models may import core; commands may import both.
``tests/test_structure.py`` enforces this.

    settings.py    YAML + command-line settings, generated from one pydantic model
    runtime.py     hardware settings, Fabric launch, logging, numerics
    batch.py       PairBatch, the model's input type
    data.py        pair files, lazily read embeddings, features, sharding, loaders
    memory.py      chunk (and tile) sizes that fit in memory, per model family
    inference.py   score a pair set across devices, memory-safely
    metrics.py     classification metrics
"""
