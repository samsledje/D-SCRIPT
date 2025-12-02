import os
import shutil
import subprocess as sp
import tempfile

from loguru import logger

from dscript.fasta import parse
from dscript.language_model import (
    embed_from_fasta,
    lm_embed,
)


class TestLanguageModel:
    @classmethod
    def setup_class(cls):
        cmd = "python setup.py install"
        proc = sp.Popen(cmd.split())
        proc.wait()

        # Create a temporary directory that will persist for the entire test class
        cls.temp_dir = tempfile.mkdtemp(prefix="dscript_lm_test_")
        logger.info(f"Created temporary directory: {cls.temp_dir}")

    @classmethod
    def teardown_class(cls):
        # Clean up the temporary directory
        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            try:
                shutil.rmtree(cls.temp_dir)
                logger.info(f"Successfully removed temporary directory: {cls.temp_dir}")
            except OSError as e:
                logger.warning(
                    f"Could not remove temporary directory {cls.temp_dir}: {e}"
                )
                # Let the OS clean it up eventually

    def test_lm_embed(self):
        _, sequences = parse("dscript/tests/test.fasta")
        for seq in sequences:
            x = lm_embed(seq)
            assert x.shape[1] == len(seq)

    def test_embed_from_fasta(self):
        """Test embedding sequences from a FASTA file"""
        output_path = f"{self.temp_dir}/test_embed.h5"
        embed_from_fasta(
            "dscript/tests/test.fasta",
            output_path,
            verbose=True,
        )

        # Verify the output file was created
        assert os.path.exists(output_path)

        # Verify embeddings are in the file
        import h5py
        with h5py.File(output_path, "r") as f:
            assert len(f.keys()) > 0

    def test_embed_from_fasta_cpu_device(self):
        """Test embedding with explicit CPU device"""
        output_path = f"{self.temp_dir}/test_embed_cpu.h5"
        embed_from_fasta(
            "dscript/tests/test.fasta",
            output_path,
            device=-1,  # Force CPU
            verbose=False,
        )

        assert os.path.exists(output_path)

    def test_lm_embed_length_consistency(self):
        """Test that embedding length matches sequence length"""
        _, sequences = parse("dscript/tests/test.fasta")
        for seq in sequences:
            x = lm_embed(seq, use_cuda=False)
            # Shape is (1, seq_len, embedding_dim)
            assert x.shape[1] == len(seq)
            assert len(x.shape) == 3

    def test_lm_embed_output_shape(self):
        """Test that embedding has correct output shape"""
        test_seq = "MKTAYIAKQRQISFVKSHFSRQ"
        x = lm_embed(test_seq, use_cuda=False)

        # Should be (batch=1, seq_len, embedding_dim=100)
        assert x.shape[0] == 1
        assert x.shape[1] == len(test_seq)
        assert x.shape[2] == 100

    def test_lm_embed_short_sequence(self):
        """Test embedding a very short sequence"""
        short_seq = "MK"
        x = lm_embed(short_seq, use_cuda=False)

        assert x.shape[1] == 2
        assert x.shape[2] == 100

    def test_lm_embed_single_amino_acid(self):
        """Test embedding a single amino acid"""
        single_aa = "M"
        x = lm_embed(single_aa, use_cuda=False)

        assert x.shape[1] == 1
        assert x.shape[2] == 100

    def test_lm_embed_returns_tensor(self):
        """Test that lm_embed returns a torch tensor"""
        import torch
        test_seq = "MKTAYIAKQR"
        x = lm_embed(test_seq, use_cuda=False)

        assert isinstance(x, torch.Tensor)

    def test_lm_embed_deterministic(self):
        """Test that lm_embed produces consistent results"""
        import torch
        test_seq = "MKTAYIAKQRQISFVKSHFSRQ"

        # Embed the same sequence twice
        x1 = lm_embed(test_seq, use_cuda=False)
        x2 = lm_embed(test_seq, use_cuda=False)

        # Results should be identical (model is in eval mode)
        assert torch.allclose(x1, x2, rtol=1e-5)

    def test_embed_from_fasta_no_verbose(self):
        """Test embedding without verbose output"""
        output_path = f"{self.temp_dir}/test_embed_quiet.h5"
        embed_from_fasta(
            "dscript/tests/test.fasta",
            output_path,
            verbose=False,
        )

        assert os.path.exists(output_path)

    def test_embed_from_fasta_creates_valid_h5(self):
        """Test that embed_from_fasta creates valid HDF5 file"""
        import h5py
        output_path = f"{self.temp_dir}/test_embed_valid.h5"

        # Parse original sequences
        names, sequences = parse("dscript/tests/test.fasta")

        embed_from_fasta(
            "dscript/tests/test.fasta",
            output_path,
            verbose=False,
        )

        # Verify HDF5 structure
        with h5py.File(output_path, "r") as f:
            # Check that all sequences are embedded
            for name, seq in zip(names, sequences):
                assert name in f
                embedding = f[name][:]
                # Check embedding dimensions
                assert embedding.shape[0] == 1
                assert embedding.shape[1] == len(seq)
                assert embedding.shape[2] == 100

    def test_embed_from_fasta_handles_existing_file(self):
        """Test that embed_from_fasta can append to existing file"""
        import h5py
        output_path = f"{self.temp_dir}/test_embed_append.h5"

        # First embedding
        embed_from_fasta(
            "dscript/tests/test.fasta",
            output_path,
            verbose=False,
        )

        # Get count of embeddings
        with h5py.File(output_path, "r") as f:
            count_before = len(f.keys())

        # Second embedding (should skip existing)
        embed_from_fasta(
            "dscript/tests/test.fasta",
            output_path,
            verbose=False,
        )

        # Count should be the same (no duplicates)
        with h5py.File(output_path, "r") as f:
            count_after = len(f.keys())

        assert count_before == count_after
