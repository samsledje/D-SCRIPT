import os
import shutil
import subprocess as sp
from Bio import SeqIO

from dscript.language_model import (
    lm_embed,
    embed_from_fasta,
)


class TestLanguageModel:
    @classmethod
    def setup_class(cls):
        cmd = "python setup.py install"
        proc = sp.Popen(cmd.split())
        proc.wait()
        os.makedirs("./tmp-dscript-testing/", exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("./tmp-dscript-testing/")

    def test_lm_embed(self):
        seqs = list(SeqIO.parse("dscript/tests/test.fasta", "fasta"))
        for seqrec in seqs:
            x = lm_embed(str(seqrec.seq))
            assert x.shape[1] == len(seqrec.seq)

    def embed_from_fasta(self):
        embed_from_fasta(
            "dscript/tests/test.fasta",
            "tmp-dscript-testing/test_embed.h5",
            verbose=True,
        )
