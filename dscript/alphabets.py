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

    def unpack(self, h, k):
        """unpack integer h into array of this alphabet with length k"""
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """retrieve byte string of length k decoded from integer h"""
        kmer = self.unpack(h, k)
        return self.decode(kmer)


DNA = Alphabet(b"ACGT")


class Uniprot21(Alphabet):
    """
    Uniprot 21 Amino Acid Encoding.

    From `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.
    """

    def __init__(self, mask=False):
        chars = b"ARNDCQEGHILKMFPSTWYVXOUBZ"
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        super().__init__(chars, encoding=encoding, mask=mask, missing=20)


class SDM12(Alphabet):
    """
    A D KER N TSQ YF LIVM C W H G P

    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2732308/#B33
    "Reduced amino acid alphabets exhibit an improved sensitivity and selectivity in fold assignment"
    Peterson et al. 2009. Bioinformatics.
    """

    def __init__(self, mask=False):
        chars = b"ADKNTYLCWHGPXERSQFIVMOUBZ"
        groups = [
            b"A",
            b"D",
            b"KERO",
            b"N",
            b"TSQ",
            b"YF",
            b"LIVM",
            b"CU",
            b"W",
            b"H",
            b"G",
            b"P",
            b"XBZ",
        ]
        groups = {c: i for i in range(len(groups)) for c in groups[i]}
        encoding = np.array([groups[c] for c in chars])
        super().__init__(chars, encoding=encoding, mask=mask)


SecStr8 = Alphabet(b"HBEGITS ")
