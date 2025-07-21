import os
import shutil
import subprocess as sp
import tempfile

from Bio import SeqIO
from loguru import logger

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
        seqs = list(SeqIO.parse("dscript/tests/test.fasta", "fasta"))
        for seqrec in seqs:
            x = lm_embed(str(seqrec.seq))
            assert x.shape[1] == len(seqrec.seq)

    def embed_from_fasta(self):
        embed_from_fasta(
            "dscript/tests/test.fasta",
            f"{self.temp_dir}/test_embed.h5",
            verbose=True,
        )
