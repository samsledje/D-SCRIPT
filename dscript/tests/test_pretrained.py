from pathlib import Path

from dscript.pretrained import (
    build_human_1,
    build_lm_1,
    get_pretrained,
    get_state_dict,
)

MODEL_VERSIONS = [
    "human_v1",  # Original D-SCRIPT Model
    "human_v2",  # Topsy-Turvy
    "lm_v1",  # Bepler & Berger 2019
]


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


def test_build_human_2():
    sd = get_state_dict("human_v2")
    build_human_1(sd)


def test_get_pretrained():
    get_pretrained("human_v1")
    get_pretrained("human_v2")
    get_pretrained("lm_v1")
