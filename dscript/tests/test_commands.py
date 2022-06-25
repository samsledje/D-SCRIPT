import os
import shutil
import subprocess as sp

DEVICE = 7


class TestCommands:
    @classmethod
    def setup_class(cls):
        cmd = "python setup.py install"
        proc = sp.Popen(cmd.split())
        proc.wait()
        os.makedirs("./tmp-dscript-testing/", exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("./tmp-dscript-testing/")

    def _run_command(self, cmd):
        proc = sp.Popen(cmd.split())
        proc.wait()
        assert not proc.returncode

    def test_embed(self):
        cmd = f"dscript embed --seqs dscript/tests/test.fasta --outfile tmp-dscript-testing/test_embed.h5 --device {DEVICE}"
        self._run_command(cmd)

    #    def test_train_with_topsy_turvy(self):
    #        cmd = "dscript train --topsy-turvy --train dscript/tests/test.csv --val dscript/tests/test.csv --embedding tmp-dscript-testing/test_embed.h5 --outfile tmp-dscript-testing/test_tt-train.log --save-prefix tmp-dscript-testing/test_train --device 0"
    #        self._run_command(cmd)

    #   def test_train_without_topsy_turvy(self):
    #       cmd = f"dscript train --train dscript/tests/test.csv --val dscript/tests/test.csv --embedding tmp-dscript-testing/test_embed.h5 --outfile tmp-dscript-testing/test_tt-train.log --save-prefix tmp-dscript-testing/test_train --device {DEVICE}"
    #       self._run_command(cmd)

    #   def test_evaluate(self):
    #       cmd = f"dscript evaluate --test dscript/tests/test.csv --embedding tmp-dscript-testing/test_embed.h5 --model tmp-dscript-testing/test_train_final.sav --outfile tmp-dscript-testing/test_evaluate --device {DEVICE}"
    #       self._run_command(cmd)

    #   def test_predict(self):
    #       cmd = f"dscript predict --seqs dscript/tests/test.fasta --pairs dscript/tests/test.csv --model tmp-dscript-testing/test_train_final.sav --outfile tmp-dscript-testing/test_predict --thresh 0.05 --device {DEVICE}"
    #       self._run_command(cmd)
