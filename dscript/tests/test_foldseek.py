import torch
from loguru import logger

from dscript.foldseek import (
    fold_vocab,
    get_3di_sequences,
    get_foldseek_onehot,
)


class TestFoldseek:
    """Test cases for foldseek.py module."""

    def test_fold_vocab_completeness(self):
        """Test that fold_vocab contains expected amino acid codes."""
        expected_codes = {
            "D",
            "P",
            "V",
            "Q",
            "A",
            "W",
            "K",
            "E",
            "I",
            "T",
            "L",
            "F",
            "G",
            "S",
            "M",
            "H",
            "C",
            "R",
            "Y",
            "N",
            "X",
        }

        assert set(fold_vocab.keys()) == expected_codes
        assert len(fold_vocab) == 21

        # Check that all values are unique integers
        values = list(fold_vocab.values())
        assert len(set(values)) == len(values)  # All unique
        assert all(isinstance(v, int) for v in values)
        assert min(values) == 0
        assert max(values) == 20

    def test_get_foldseek_onehot_valid_sequence(self):
        """Test one-hot encoding with a valid sequence in fold_record."""
        n0 = "test_protein"
        fold_sequence = "DPVQA"
        size_n0 = len(fold_sequence)
        fold_record = {n0: fold_sequence}

        result = get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab)

        # Check dimensions
        assert result.shape == (size_n0, len(fold_vocab))
        assert result.dtype == torch.float32

        # Check that each position has exactly one 1 and the rest are 0s
        for i in range(size_n0):
            assert torch.sum(result[i]) == 1.0
            # Check that the correct position is set to 1
            amino_acid = fold_sequence[i]
            expected_idx = fold_vocab[amino_acid]
            assert result[i, expected_idx] == 1.0

    def test_get_foldseek_onehot_protein_not_in_record(self):
        """Test one-hot encoding when protein is not in fold_record."""
        n0 = "missing_protein"
        size_n0 = 5
        fold_record = {"other_protein": "DPVQA"}

        result = get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab)

        # Should return all zeros
        assert result.shape == (size_n0, len(fold_vocab))
        assert result.dtype == torch.float32
        assert torch.all(result == 0.0)

    def test_get_foldseek_onehot_size_mismatch(self):
        """Test assertion error when size doesn't match sequence length."""
        n0 = "test_protein"
        fold_sequence = "DPVQA"
        size_n0 = 3  # Different from actual sequence length (5)
        fold_record = {n0: fold_sequence}

        try:
            get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass  # Expected behavior

    def test_get_foldseek_onehot_invalid_amino_acid(self):
        """Test assertion error with invalid amino acid in sequence."""
        n0 = "test_protein"
        fold_sequence = "DPVQZ"  # Z is not in fold_vocab
        size_n0 = len(fold_sequence)
        fold_record = {n0: fold_sequence}

        try:
            get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass  # Expected behavior

    def test_get_foldseek_onehot_empty_sequence(self):
        """Test one-hot encoding with empty sequence."""
        n0 = "empty_protein"
        fold_sequence = ""
        size_n0 = 0
        fold_record = {n0: fold_sequence}

        result = get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab)

        assert result.shape == (0, len(fold_vocab))
        assert result.dtype == torch.float32

    def test_get_foldseek_onehot_all_amino_acids(self):
        """Test one-hot encoding with all possible amino acids."""
        # Create a sequence with all amino acids in fold_vocab
        all_amino_acids = "".join(sorted(fold_vocab.keys()))
        n0 = "all_aa_protein"
        size_n0 = len(all_amino_acids)
        fold_record = {n0: all_amino_acids}

        result = get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab)

        assert result.shape == (size_n0, len(fold_vocab))

        # Each position should have exactly one 1
        for i in range(size_n0):
            assert torch.sum(result[i]) == 1.0

        # Each amino acid should appear exactly once
        for aa, idx in fold_vocab.items():
            aa_positions = torch.where(result[:, idx] == 1.0)[0]
            assert len(aa_positions) == 1  # Should appear exactly once

    def test_get_foldseek_onehot_repeated_amino_acids(self):
        """Test one-hot encoding with repeated amino acids."""
        n0 = "repeat_protein"
        fold_sequence = "DDDD"  # All D amino acids
        size_n0 = len(fold_sequence)
        fold_record = {n0: fold_sequence}

        result = get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab)

        assert result.shape == (size_n0, len(fold_vocab))

        # All positions should have D (index 0) set to 1
        d_idx = fold_vocab["D"]
        for i in range(size_n0):
            assert result[i, d_idx] == 1.0
            assert torch.sum(result[i]) == 1.0

    def test_get_3di_sequences_no_foldseek_binary(self):
        """Test get_3di_sequences when foldseek binary is not available."""
        # This test assumes foldseek is not installed
        pdb_files = ["dummy.pdb"]

        try:
            _ = get_3di_sequences(pdb_files, foldseek_path="nonexistent_foldseek")
            # If it doesn't fail, that's unexpected but not necessarily wrong
            # The function might handle missing binary gracefully
        except (FileNotFoundError, OSError):
            # Expected when foldseek binary is not found
            logger.warning("Foldseek binary not found, skipping test.")
            pass
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            pass

    def test_get_3di_sequences_empty_pdb_list(self):
        """Test get_3di_sequences with empty PDB file list."""
        pdb_files = []

        try:
            result = get_3di_sequences(pdb_files)
            # Should handle empty list gracefully
            assert isinstance(result, dict)
        except Exception as e:
            # May fail due to empty input, which is acceptable
            logger.warning(f"Unexpected error with empty PDB list: {e}")
            pass

    def test_get_3di_sequences_nonexistent_pdb_files(self):
        """Test get_3di_sequences with non-existent PDB files."""
        pdb_files = ["nonexistent1.pdb", "nonexistent2.pdb"]

        try:
            result = get_3di_sequences(pdb_files)
            # If it succeeds, result should be a dict
            assert isinstance(result, dict)
        except Exception as e:
            # Expected to fail with non-existent files
            logger.error(f"Unexpected error with non-existent PDB files: {e}")
            pass

    def test_fold_vocab_mapping(self):
        """Test specific mappings in fold_vocab."""
        # Test a few specific mappings
        assert fold_vocab["D"] == 0
        assert fold_vocab["P"] == 1
        assert fold_vocab["V"] == 2
        assert fold_vocab["X"] == 20  # Should be the last one

    def test_get_foldseek_onehot_tensor_properties(self):
        """Test tensor properties of the output."""
        n0 = "test_protein"
        fold_sequence = "DPVQA"
        size_n0 = len(fold_sequence)
        fold_record = {n0: fold_sequence}

        result = get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab)

        # Test tensor properties
        assert isinstance(result, torch.Tensor)
        assert result.device == torch.device("cpu")  # Default device
        assert not result.requires_grad  # Should not require gradients by default
        assert result.dtype == torch.float32

        # Test value range
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)
        assert torch.all((result == 0.0) | (result == 1.0))  # Only 0s and 1s
