import os
import shutil
import subprocess as sp
import tempfile

import h5py
import torch
from loguru import logger


class TestCommands:
    @classmethod
    def setup_class(cls):
        cmd = "python setup.py install"
        proc = sp.Popen(cmd.split())
        proc.wait()

        # Create a temporary directory that will persist for the entire test class
        cls.temp_dir = tempfile.mkdtemp(prefix="dscript_test_")
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

    def _run_command(self, cmd):
        proc = sp.Popen(cmd.split())
        proc.wait()
        assert not proc.returncode

    def test_embed(self):
        cmd = f"dscript embed --seqs dscript/tests/test.fasta --outfile {self.temp_dir}/test_embed.h5"
        self._run_command(cmd)
        assert os.path.exists(f"{self.temp_dir}/test_embed.h5"), (
            "Embedding file not created."
        )
        with h5py.File(f"{self.temp_dir}/test_embed.h5", "r") as f:
            klist = list(f.keys())
            assert len(klist) == 3, "Expected 3 keys in the embedding file."

    def test_train_with_topsy_turvy(self):
        cmd = f"dscript train --topsy-turvy --train dscript/tests/test.tsv --test dscript/tests/test.tsv --embedding {self.temp_dir}/test_embed.h5 --outfile {self.temp_dir}/test_tt-train.log --save-prefix {self.temp_dir}/test_tt-train"
        self._run_command(cmd)
        assert os.path.exists(f"{self.temp_dir}/test_tt-train_final.sav"), (
            "Model file not created."
        )

    def test_train_without_topsy_turvy(self):
        cmd = f"dscript train --train dscript/tests/test.tsv --test dscript/tests/test.tsv --embedding {self.temp_dir}/test_embed.h5 --outfile {self.temp_dir}/test-train.log --save-prefix {self.temp_dir}/test_train"
        self._run_command(cmd)
        assert os.path.exists(f"{self.temp_dir}/test_train_final.sav"), (
            "Model file not created."
        )

    def test_evaluate(self):
        cmd = f"dscript evaluate --test dscript/tests/test.tsv --embeddings {self.temp_dir}/test_embed.h5 --model {self.temp_dir}/test_train_final.sav --outfile {self.temp_dir}/test_evaluate"
        self._run_command(cmd)

    def test_predict_one_gpu(self):
        if torch.cuda.is_available():
            cmd = f"dscript predict --pairs dscript/tests/test.tsv --embeddings {self.temp_dir}/test_embed.h5 --model {self.temp_dir}/test_train_final.sav --outfile {self.temp_dir}/test_predict --thresh 0.05 --device 0"
            self._run_command(cmd)
            with open(f"{self.temp_dir}/test_predict.tsv") as f:
                lines = f.readlines()
                assert len(lines) > 0, "Output file is empty."
                assert len(lines) == 3, "Expected 3 lines in the output file."
        else:
            logger.warning("CUDA is not available, skipping GPU prediction test.")

    def test_predict_all_gpu(self):
        if torch.cuda.is_available():
            cmd = f"dscript predict --pairs dscript/tests/test.tsv --embeddings {self.temp_dir}/test_embed.h5 --model {self.temp_dir}/test_train_final.sav --outfile {self.temp_dir}/test_predict --thresh 0.05 --device all"
            self._run_command(cmd)
            with open(f"{self.temp_dir}/test_predict.tsv") as f:
                lines = f.readlines()
                assert len(lines) > 0, "Output file is empty."
                assert len(lines) == 3, "Expected 3 lines in the output file."
        else:
            logger.warning("CUDA is not available, skipping GPU prediction test.")

    def test_predict_on_cpu(self):
        cmd = f"dscript predict --pairs dscript/tests/test.tsv --embeddings {self.temp_dir}/test_embed.h5 --model {self.temp_dir}/test_train_final.sav --outfile {self.temp_dir}/test_predict --thresh 0.05 --device cpu"
        self._run_command(cmd)
        with open(f"{self.temp_dir}/test_predict.tsv") as f:
            lines = f.readlines()
            assert len(lines) > 0, "Output file is empty."
            assert len(lines) == 3, "Expected 3 lines in the output file."

    def test_predict_bipartite(self):
        if torch.cuda.is_available():
            cmd = f"dscript predict_bipartite --protA dscript/tests/test.bipartite.txt --protB dscript/tests/test.bipartite.txt --embedA {self.temp_dir}/test_embed.h5 --model {self.temp_dir}/test_train_final.sav --outfile {self.temp_dir}/test_bipartite --thresh 0.05 --device 0"
            self._run_command(cmd)
            with open(f"{self.temp_dir}/test_bipartite.tsv") as f:
                lines = f.readlines()
                assert len(lines) > 0, "Output file is empty."
                assert len(lines) == 9, "Expected 9 lines in the output file."
        else:
            logger.warning("CUDA is not available, skipping bipartite prediction test.")

    def test_predict_bipartite_cpu(self):
        cmd = f"dscript predict_bipartite --protA dscript/tests/test.bipartite.txt --protB dscript/tests/test.bipartite.txt --embedA {self.temp_dir}/test_embed.h5 --model {self.temp_dir}/test_train_final.sav --outfile {self.temp_dir}/test_bipartite_cpu --thresh 0.05 --device cpu"
        self._run_command(cmd)
        with open(f"{self.temp_dir}/test_bipartite_cpu.tsv") as f:
            lines = f.readlines()
            assert len(lines) > 0, "Output file is empty."
            assert len(lines) == 9, "Expected 9 lines in the output file."

    def test_predict_serial(self):
        cmd = f"dscript predict_serial --pairs dscript/tests/test.tsv --embeddings {self.temp_dir}/test_embed.h5 --model {self.temp_dir}/test_train_final.sav --outfile {self.temp_dir}/test_serial --thresh 0.05"
        self._run_command(cmd)
