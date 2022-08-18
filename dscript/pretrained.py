from functools import wraps, partial
import os
import os.path
import sys

import torch

from .models.contact import ContactCNN
from .models.embedding import FullyConnectedEmbed, SkipLSTM
from .models.interaction import ModelInteraction
from .utils import log


def build_lm_1(state_dict_path):
    """
    :meta private:
    """
    model = SkipLSTM(21, 100, 1024, 3)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_human_1(state_dict_path):
    """
    :meta private:
    """
    embModel = FullyConnectedEmbed(6165, 100, 0.5)
    conModel = ContactCNN(100, 50, 7)
    model = ModelInteraction(
        embModel,
        conModel,
        use_cuda=True,
        do_w=True,
        do_pool=True,
        do_sigmoid=True,
        pool_size=9,
    )
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


VALID_MODELS = {
    "human_v1": build_human_1,  # Original D-SCRIPT
    "human_v2": build_human_1,  # Topsy-Turvy
    "lm_v1": build_lm_1,  # Bepler & Berger 2019
}

STATE_DICT_BASENAME = "dscript_{version}.pt"

ROOT_URL = "http://cb.csail.mit.edu/cb/dscript/data/models"


def get_state_dict_path(version: str) -> str:
    state_dict_basedir = os.path.dirname(os.path.realpath(__file__))
    state_dict_fullname = (
        f"{state_dict_basedir}/{STATE_DICT_BASENAME.format(version=version)}"
    )
    return state_dict_fullname


def get_state_dict(version="human_v2", verbose=True):
    """
    Download a pre-trained model if not already exists on local device.

    :param version: Version of trained model to download [default: human_1]
    :type version: str
    :param verbose: Print model download status on stdout [default: True]
    :type verbose: bool
    :return: Path to state dictionary for pre-trained language model
    :rtype: str
    """
    state_dict_fullname = get_state_dict_path(version)
    state_dict_url = (
        f"{ROOT_URL}/{STATE_DICT_BASENAME.format(version=version)}"
    )
    if not os.path.exists(state_dict_fullname):
        try:
            import shutil
            import urllib.request

            if verbose:
                log(f"Downloading model {version} from {state_dict_url}...")
            with urllib.request.urlopen(state_dict_url) as response, open(
                state_dict_fullname, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            log("Unable to download model - {}".format(e))
            sys.exit(1)
    return state_dict_fullname


def retry(retry_count: int):
    def decorate(func):
        @wraps(func)
        def retry_wrapper(*args, **kwargs):
            attempt = 0
            if len(args):
                version = args[0]
            elif "version" in kwargs:
                version = kwargs["version"]
            else:
                version = func.__defaults__[0]
            while attempt < retry_count:
                try:
                    result = func(*args, **kwargs)
                    return result
                except RuntimeError as e:
                    log(
                        f"\033[93mLoading {version} from disk failed. Retrying download attempt: {attempt + 1}\033[0m"
                    )
                    if e.args[0].startswith("unexpected EOF"):
                        state_dict_fullname = get_state_dict_path(version)
                        if os.path.exists(state_dict_fullname):
                            os.remove(state_dict_fullname)
                    else:
                        raise e
                attempt += 1
            raise Exception(f"Failed to download {version}")

        return retry_wrapper

    return decorate


@retry(3)
def get_pretrained(version="human_v2"):
    """
    Get pre-trained model object.

    Currently Available Models
    ==========================

    See the `documentation <https://d-script.readthedocs.io/en/main/data.html#trained-models>`_ for most up-to-date list.

    - ``lm_v1`` - Language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.
    - ``human_v1`` - Human trained model from D-SCRIPT manuscript.
    - ``human_v2`` - Human trained model from Topsy-Turvy manuscript.

    Default: ``human_v2``

    :param version: Version of pre-trained model to get
    :type version: str
    :return: Pre-trained model
    :rtype: dscript.models.*
    """
    if version not in VALID_MODELS:
        raise ValueError("Model {} does not exist".format(version))

    state_dict_path = get_state_dict(version)
    return VALID_MODELS[version](state_dict_path)
