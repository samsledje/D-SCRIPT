import os
import shutil
import subprocess as sp


def test_with_topsy_turvy():
    os.makedirs("./tmp-dscript-testing/", exist_ok=True)

    cmd = "dscript train --topsy-turvy --train dscript/tests/test.csv --test dscript/tests/test.csv --embedding dscript/tests/test.h5 --outfile tmp-dscript-testing/test_tt-train --save-prefix tmp-dscript-testing/test_tt-train --device 0"
    proc = sp.Popen(cmd.split())
    proc.wait()
    assert not proc.returncode

    shutil.rmtree("./tmp-dscript-testing/")


def test_without_topsy_turvy():
    os.makedirs("./tmp-dscript-testing/", exist_ok=True)

    cmd = "dscript train --train dscript/tests/test.csv --test dscript/tests/test.csv --embedding dscript/tests/test.h5 --outfile tmp-dscript-testing/test_tt-train --save-prefix tmp-dscript-testing/test_tt-train --device 0"
    proc = sp.Popen(cmd.split())
    proc.wait()
    assert not proc.returncode

    shutil.rmtree("./tmp-dscript-testing/")
