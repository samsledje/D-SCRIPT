from __future__ import print_function, division

import numpy as np


class Alphabet:
    """
    From `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_
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
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """
        Decode index array :math:`x` to byte string of this alphabet
        """
        string = self.chars[x]
        return string.tobytes()

    def unpack(self, h, k):
        """
        Unpack integer :math:`h` into array of this alphabet with length :math:`k`
        """
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """
        Retrieve byte string of length :math:`k` decoded from integer :math:`h`
        """
        kmer = self.unpack(h, k)
        return self.decode(kmer)

DNA = Alphabet(b"ACGT")
class Uniprot21(Alphabet):
    """
    Uniprot 21 Amino Acid Encoding

    From `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_
    """
    def __init__(self, mask=False):
        chars = alphabet = b"ARNDCQEGHILKMFPSTWYVXOUBZ"
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        super(Uniprot21, self).__init__(chars, encoding=encoding, mask=mask, missing=20)
