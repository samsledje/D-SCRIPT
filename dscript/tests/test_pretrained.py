import dscript
from pathlib import Path

from dscript.pretrained import (
    build_human_1,
    build_lm_1,
    get_pretrained,
    get_state_dict,
)

MODEL_VERSIONS = [
    "human_v1",
    "lm_v1",
]

print(dscript.__version__)


def test_get_state_dict():
    for mv in MODEL_VERSIONS:
        sd = get_state_dict(mv, verbose=True)
        assert Path(
            sd
        ).exists(), f"Path {sd} was not downloaded or does not exist"


def test_build_lm_1():
    sd = get_state_dict("lm_v1")
    build_lm_1(sd)


def test_build_human_1():
    sd = get_state_dict("human_v1")
    build_human_1(sd)


def test_get_pretrained():
    get_pretrained("human_v1")
    get_pretrained("lm_v1")
