"""
Batched model: no BatchNorm, so pairs are padded and run together, and the
contact map is computed in row tiles. Not weight-compatible with the serial model.

    embedding.py     Projection: residue-level projection
    contact.py       PairFeatures, ContactHead: pair features and the contact map
    interaction.py   DScriptBatched: the full model, tiling, pooling, save/load
"""

from .contact import ContactHead, PairFeatures
from .embedding import Projection
from .interaction import DScriptBatched

__all__ = ["ContactHead", "DScriptBatched", "PairFeatures", "Projection"]
