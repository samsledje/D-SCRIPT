from __future__ import print_function, division

import numpy as np


class Alphabet:
    """
    From `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param chars: List of characters in alphabet
    :type chars: byte str
    :param encoding: Mapping of characters to numbers [default: encoding]
    :type encoding: np.ndarray
    :param mask: Set encoding mask [default: False]
    :type mask: bool
    :param missing: Number to use for a value outside the alphabet [default: 255]
    :type missing: int
    """

    def __init__(self, chars, encoding=None, mask=False, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
        self.mask = mask
        if mask:
            self.size -= 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """
        Encode a byte string into alphabet indices

        :param x: Amino acid string
        :type x: byte str
        :return: Numeric encoding
        :rtype: np.ndarray
        """
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """
        Decode numeric encoding to byte string of this alphabet

        :param x: Numeric encoding
        :type x: np.ndarray
        :return: Amino acid string
        :rtype: byte str
        """
        string = self.chars[x]
        return string.tobytes()


class Uniprot21(Alphabet):
    """
    Uniprot 21 Amino Acid Encoding.

    From `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.
    """

    def __init__(self, mask=False):
        chars = b"ARNDCQEGHILKMFPSTWYVXOUBZ"
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        super(Uniprot21, self).__init__(
            chars, encoding=encoding, mask=mask, missing=20
        )
