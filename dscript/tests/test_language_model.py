"""
Tests for language model embedding functionality in dscript.language_model
"""

from unittest.mock import Mock, patch

import h5py
import pytest
import torch

from dscript.fasta import parse
from dscript.language_model import (
    embed_from_fasta,
    lm_embed,
)


class TestLanguageModelUnit:
    """Unit tests with mocked model"""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that behaves like the real one"""
        model = Mock()
        model.eval = Mock()
        model.cuda = Mock(return_value=model)
        model.cpu = Mock(return_value=model)

        # Mock the proj layer
        model.proj = Mock()
        model.proj.weight = torch.randn(6165, 6165)
        model.proj.bias = torch.zeros(6165)

        # Mock transform to return realistic embeddings
        def mock_transform(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            # Return (batch, seq_len, embedding_dim=6165)
            return torch.randn(batch_size, seq_len, 6165)

        model.transform = Mock(side_effect=mock_transform)
        return model

    @patch("dscript.language_model.get_pretrained")
    def test_lm_embed_shape(self, mock_get_pretrained, mock_model):
        """Test that lm_embed returns correct shape"""
        mock_get_pretrained.return_value = mock_model

        test_seq = "MKTAYIAKQRQISFVKSHFSRQ"
        x = lm_embed(test_seq, use_cuda=False)

        # Should be (batch=1, seq_len, embedding_dim=6165)
        assert x.shape[0] == 1
        assert x.shape[1] == len(test_seq)
        assert x.shape[2] == 6165

    @patch("dscript.language_model.get_pretrained")
    def test_lm_embed_returns_tensor(self, mock_get_pretrained, mock_model):
        """Test that lm_embed returns a torch tensor"""
        mock_get_pretrained.return_value = mock_model

        test_seq = "MKTAYIAKQR"
        x = lm_embed(test_seq, use_cuda=False)

        assert isinstance(x, torch.Tensor)

    @patch("dscript.language_model.get_pretrained")
    def test_lm_embed_short_sequence(self, mock_get_pretrained, mock_model):
        """Test embedding a very short sequence"""
        mock_get_pretrained.return_value = mock_model

        short_seq = "MK"
        x = lm_embed(short_seq, use_cuda=False)

        assert x.shape[1] == 2
        assert x.shape[2] == 6165

    @patch("dscript.language_model.get_pretrained")
    def test_lm_embed_single_amino_acid(self, mock_get_pretrained, mock_model):
        """Test embedding a single amino acid"""
        mock_get_pretrained.return_value = mock_model

        single_aa = "M"
        x = lm_embed(single_aa, use_cuda=False)

        assert x.shape[1] == 1
        assert x.shape[2] == 6165

    @patch("dscript.language_model.get_pretrained")
    def test_embed_from_fasta_creates_h5(self, mock_get_pretrained, mock_model, tmp_path):
        """Test that embed_from_fasta creates HDF5 file"""
        mock_get_pretrained.return_value = mock_model

        output_path = tmp_path / "test_embed.h5"

        embed_from_fasta(
            "dscript/tests/test.fasta",
            str(output_path),
            device=-1,  # Force CPU
            verbose=False,
        )

        # Verify the output file was created
        assert output_path.exists()

        # Verify it's a valid HDF5 file
        with h5py.File(output_path, "r") as f:
            assert len(f.keys()) > 0

    @patch("dscript.language_model.get_pretrained")
    def test_embed_from_fasta_correct_names(
        self, mock_get_pretrained, mock_model, tmp_path
    ):
        """Test that embed_from_fasta uses correct sequence names"""
        mock_get_pretrained.return_value = mock_model

        output_path = tmp_path / "test_embed.h5"

        # Parse original sequences to get names
        names, _ = parse("dscript/tests/test.fasta")

        embed_from_fasta(
            "dscript/tests/test.fasta",
            str(output_path),
            device=-1,
            verbose=False,
        )

        # Verify all sequence names are in the output
        with h5py.File(output_path, "r") as f:
            for name in names:
                assert name in f

    @patch("dscript.language_model.get_pretrained")
    def test_embed_from_fasta_skips_existing(
        self, mock_get_pretrained, mock_model, tmp_path
    ):
        """Test that embed_from_fasta skips existing embeddings"""
        mock_get_pretrained.return_value = mock_model

        output_path = tmp_path / "test_embed.h5"

        # First embedding
        embed_from_fasta(
            "dscript/tests/test.fasta",
            str(output_path),
            device=-1,
            verbose=False,
        )

        # Get count of embeddings
        with h5py.File(output_path, "r") as f:
            count_before = len(f.keys())

        # Second embedding (should skip existing)
        embed_from_fasta(
            "dscript/tests/test.fasta",
            str(output_path),
            device=-1,
            verbose=False,
        )

        # Count should be the same (no duplicates)
        with h5py.File(output_path, "r") as f:
            count_after = len(f.keys())

        assert count_before == count_after

    @patch("dscript.language_model.get_pretrained")
    def test_embed_from_fasta_cpu_device(self, mock_get_pretrained, mock_model, tmp_path):
        """Test embedding with explicit CPU device"""
        mock_get_pretrained.return_value = mock_model

        output_path = tmp_path / "test_embed_cpu.h5"

        embed_from_fasta(
            "dscript/tests/test.fasta",
            str(output_path),
            device=-1,  # Force CPU
            verbose=False,
        )

        assert output_path.exists()

    @patch("dscript.language_model.get_pretrained")
    @patch("dscript.language_model.log")
    def test_embed_from_fasta_verbose_output(
        self, mock_log, mock_get_pretrained, mock_model, tmp_path
    ):
        """Test that verbose mode produces log output"""
        mock_get_pretrained.return_value = mock_model

        output_path = tmp_path / "test_embed.h5"

        embed_from_fasta(
            "dscript/tests/test.fasta",
            str(output_path),
            device=-1,
            verbose=True,
        )

        # Verbose mode should call log
        assert mock_log.called


@pytest.mark.slow
class TestLanguageModelIntegration:
    """Integration tests that use the real model (marked as slow)"""

    def test_lm_embed_real(self):
        """Test lm_embed with real model (slow)"""
        # This test actually loads the model and runs inference
        test_seq = "MKTAYIAK"
        x = lm_embed(test_seq, use_cuda=False)

        assert x.shape[0] == 1
        assert x.shape[1] == len(test_seq)
        assert x.shape[2] == 6165
        assert isinstance(x, torch.Tensor)

    def test_embed_from_fasta_real(self, tmp_path):
        """Test embed_from_fasta with real model (slow)"""
        output_path = tmp_path / "test_embed_real.h5"

        embed_from_fasta(
            "dscript/tests/test.fasta",
            str(output_path),
            device=-1,
            verbose=False,
        )

        assert output_path.exists()

        # Verify HDF5 structure
        names, sequences = parse("dscript/tests/test.fasta")
        with h5py.File(output_path, "r") as f:
            for name, seq in zip(names, sequences):
                assert name in f
                embedding = f[name][:]
                assert embedding.shape[0] == 1
                assert embedding.shape[1] == len(seq)
                assert embedding.shape[2] == 6165
