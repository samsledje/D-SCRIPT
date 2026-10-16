"""
The original D-SCRIPT model (contact CNN with BatchNorm, per-pair forward),
the Bepler & Berger language model, and the batch type they use.

Kept only because embedding, prediction, evaluation and the published
pretrained models still depend on them. Training uses ``dscript.models``.
Delete this package once those commands have been rebuilt on ``dscript.core``.
"""
