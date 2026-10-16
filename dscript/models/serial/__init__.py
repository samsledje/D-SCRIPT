"""
Serial model: the original D-SCRIPT architecture. Pairs run one at a time;
BatchNorm takes its training statistics from each pair's whole contact map.

    embedding.py     FullyConnectedEmbed: residue-level projection
    contact.py       FullyConnected, ContactCNN: pair features and the contact map
    interaction.py   DScriptSerial: the full model, pooling, save/load
"""

from .contact import ContactCNN, FullyConnected
from .embedding import FullyConnectedEmbed
from .interaction import DScriptSerial

__all__ = ["ContactCNN", "DScriptSerial", "FullyConnected", "FullyConnectedEmbed"]
