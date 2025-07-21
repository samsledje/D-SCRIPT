import numpy as np

from dscript.alphabets import DNA, SDM12, Alphabet, SecStr8, Uniprot21


class TestAlphabet:
    """Test cases for Alphabet class."""

    def test_alphabet_init_default(self):
        """Test basic Alphabet initialization."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        assert len(alphabet) == 4
        assert alphabet.size == 4
        assert not alphabet.mask
        assert alphabet.chars.dtype == np.uint8

    def test_alphabet_init_with_encoding(self):
        """Test Alphabet initialization with custom encoding."""
        chars = b"ACGT"
        encoding = np.array([3, 2, 1, 0])
        alphabet = Alphabet(chars, encoding=encoding)

        assert len(alphabet) == 4
        assert alphabet.size == 4

    def test_alphabet_init_with_mask(self):
        """Test Alphabet initialization with mask=True."""
        chars = b"ACGT"
        alphabet = Alphabet(chars, mask=True)

        assert len(alphabet) == 3  # size - 1 when masked
        assert alphabet.mask

    def test_alphabet_getitem(self):
        """Test indexing into alphabet."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        assert alphabet[0] == "A"
        assert alphabet[1] == "C"
        assert alphabet[2] == "G"
        assert alphabet[3] == "T"

    def test_alphabet_encode_basic(self):
        """Test encoding byte strings."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        # Test encoding single characters
        encoded = alphabet.encode(b"A")
        assert encoded[0] == 0

        encoded = alphabet.encode(b"ACGT")
        np.testing.assert_array_equal(encoded, [0, 1, 2, 3])

    def test_alphabet_encode_missing_char(self):
        """Test encoding with missing character uses default missing value."""
        chars = b"ACG"  # Missing T
        alphabet = Alphabet(chars, missing=99)

        encoded = alphabet.encode(b"ACGT")
        expected = [0, 1, 2, 99]  # T should be encoded as missing value
        np.testing.assert_array_equal(encoded, expected)

    def test_alphabet_decode(self):
        """Test decoding numeric encoding back to byte string."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        # Encode then decode should give original
        encoded = alphabet.encode(b"ACGT")
        decoded = alphabet.decode(encoded)
        assert decoded == b"ACGT"

    def test_alphabet_unpack(self):
        """Test unpacking integer to array."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        # Test unpacking - this tests the mathematical conversion
        result = alphabet.unpack(0, 3)  # unpack 0 into length 3
        expected = np.array([0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

        # Test with non-zero value
        result = alphabet.unpack(5, 2)  # 5 in base 4 is [1, 1]
        expected = np.array([1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_alphabet_get_kmer(self):
        """Test getting k-mer from integer."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        # Test getting k-mer
        kmer = alphabet.get_kmer(0, 2)
        assert kmer == b"AA"  # 0 unpacked to [0,0] then decoded to "AA"

        # Test another value
        kmer = alphabet.get_kmer(5, 2)  # 5 in base 4 = [1,1] = "CC"
        assert kmer == b"CC"


class TestDNA:
    """Test cases for DNA alphabet."""

    def test_dna_alphabet(self):
        """Test DNA alphabet basic functionality."""
        assert len(DNA) == 4
        assert DNA[0] == "A"
        assert DNA[1] == "C"
        assert DNA[2] == "G"
        assert DNA[3] == "T"

        # Test encoding
        encoded = DNA.encode(b"ACGT")
        np.testing.assert_array_equal(encoded, [0, 1, 2, 3])

        # Test decoding
        decoded = DNA.decode(encoded)
        assert decoded == b"ACGT"


class TestUniprot21:
    """Test cases for Uniprot21 alphabet."""

    def test_uniprot21_init_default(self):
        """Test Uniprot21 initialization."""
        alphabet = Uniprot21()
        assert len(alphabet) == 21
        assert not alphabet.mask

    def test_uniprot21_init_with_mask(self):
        """Test Uniprot21 initialization with mask."""
        alphabet = Uniprot21(mask=True)
        assert len(alphabet) == 20  # 21 - 1
        assert alphabet.mask

    def test_uniprot21_synonym_encoding(self):
        """Test that synonyms are encoded correctly."""
        alphabet = Uniprot21()

        # Test that synonyms map to correct values
        # 'O' should map to same as 'K' (index 11)
        o_encoded = alphabet.encode(b"O")[0]
        k_encoded = alphabet.encode(b"K")[0]
        assert o_encoded == k_encoded

        # 'U' should map to same as 'C' (index 4)
        u_encoded = alphabet.encode(b"U")[0]
        c_encoded = alphabet.encode(b"C")[0]
        assert u_encoded == c_encoded

        # 'B' and 'Z' should map to same as 'X' (index 20)
        b_encoded = alphabet.encode(b"B")[0]
        z_encoded = alphabet.encode(b"Z")[0]
        x_encoded = alphabet.encode(b"X")[0]
        assert b_encoded == x_encoded
        assert z_encoded == x_encoded

    def test_uniprot21_standard_amino_acids(self):
        """Test encoding of standard amino acids."""
        alphabet = Uniprot21()

        # Test some standard amino acids
        standard_aas = b"ARNDCQEGHILKMFPSTWYVX"
        encoded = alphabet.encode(standard_aas)

        # Should be encoded as sequential numbers 0-20
        expected = np.arange(21)
        np.testing.assert_array_equal(encoded, expected)


class TestSDM12:
    """Test cases for SDM12 alphabet."""

    def test_sdm12_init_default(self):
        """Test SDM12 initialization."""
        alphabet = SDM12()
        assert len(alphabet) == 13
        assert not alphabet.mask

    def test_sdm12_init_with_mask(self):
        """Test SDM12 initialization with mask."""
        alphabet = SDM12(mask=True)
        assert len(alphabet) == 12  # 13 - 1
        assert alphabet.mask

    def test_sdm12_grouping(self):
        """Test that amino acids are grouped correctly."""
        alphabet = SDM12()

        # Test that amino acids in same group have same encoding
        # K, E, R, O should all map to group 2
        k_encoded = alphabet.encode(b"K")[0]
        e_encoded = alphabet.encode(b"E")[0]
        r_encoded = alphabet.encode(b"R")[0]
        o_encoded = alphabet.encode(b"O")[0]

        assert k_encoded == e_encoded == r_encoded == o_encoded

        # T, S, Q should all map to group 4
        t_encoded = alphabet.encode(b"T")[0]
        s_encoded = alphabet.encode(b"S")[0]
        q_encoded = alphabet.encode(b"Q")[0]

        assert t_encoded == s_encoded == q_encoded

        # Y, F should map to group 5
        y_encoded = alphabet.encode(b"Y")[0]
        f_encoded = alphabet.encode(b"F")[0]

        assert y_encoded == f_encoded

        # L, I, V, M should map to group 6
        l_encoded = alphabet.encode(b"L")[0]
        i_encoded = alphabet.encode(b"I")[0]
        v_encoded = alphabet.encode(b"V")[0]
        m_encoded = alphabet.encode(b"M")[0]

        assert l_encoded == i_encoded == v_encoded == m_encoded


class TestSecStr8:
    """Test cases for SecStr8 alphabet."""

    def test_secstr8_alphabet(self):
        """Test SecStr8 alphabet basic functionality."""
        assert len(SecStr8) == 8

        # Test that it contains expected characters
        chars = b"HBEGITS "
        for i, char in enumerate(chars):
            assert SecStr8[i] == chr(char)

        # Test encoding
        encoded = SecStr8.encode(chars)
        expected = np.arange(8)
        np.testing.assert_array_equal(encoded, expected)

        # Test decoding
        decoded = SecStr8.decode(encoded)
        assert decoded == chars


class TestAlphabetEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_alphabet(self):
        """Test behavior with empty alphabet."""
        chars = b""
        alphabet = Alphabet(chars)
        assert len(alphabet) == 0

    def test_large_missing_value(self):
        """Test with large missing value."""
        chars = b"AC"
        alphabet = Alphabet(chars, missing=200)

        # Encode character not in alphabet
        encoded = alphabet.encode(b"T")
        assert encoded[0] == 200

    def test_encode_empty_string(self):
        """Test encoding empty byte string."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        encoded = alphabet.encode(b"")
        assert len(encoded) == 0

    def test_decode_empty_array(self):
        """Test decoding empty array."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        decoded = alphabet.decode(np.array([], dtype=np.uint8))
        assert decoded == b""

    def test_unpack_zero_length(self):
        """Test unpacking with k=0."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        result = alphabet.unpack(5, 0)
        assert len(result) == 0

    def test_get_kmer_zero_length(self):
        """Test getting k-mer with k=0."""
        chars = b"ACGT"
        alphabet = Alphabet(chars)

        kmer = alphabet.get_kmer(5, 0)
        assert kmer == b""
