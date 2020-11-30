import os, sys
import torch

from .models.embedding import FullyConnectedEmbed, SkipLSTM
from .models.contact import ContactCNN
from .models.interaction import ModelInteraction


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
    model = ModelInteraction(embModel, conModel, use_W=True, pool_size=9)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


VALID_MODELS = {
        "lm_v1": build_lm_1,
        "human_v1": build_human_1
        }


def get_state_dict(version="human_v1", verbose=True):
    """
    Download a pre-trained model if not already exists on local device.

    :param version: Version of trained model to download [default: human_1]
    :type version: str
    :param verbose: Print model download status on stdout [default: True]
    :type verbose: bool
    :return: Path to state dictionary for pre-trained language model
    :rtype: str
    """
    state_dict_basename = f"dscript_{version}.pt"
    state_dict_basedir = os.path.dirname(os.path.realpath(__file__))
    state_dict_fullname = f"{state_dict_basedir}/{state_dict_basename}"
    state_dict_url = f"http://cb.csail.mit.edu/cb/dscript/data/models/{state_dict_basename}"
    if not os.path.exists(state_dict_fullname):
        try:
            import urllib.request
            import shutil
            if verbose: print(f"Downloading model {version} from {state_dict_url}...")
            with urllib.request.urlopen(state_dict_url) as response, open(state_dict_fullname, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            print("Unable to download model - {}".format(e))
            sys.exit(1)
    return state_dict_fullname


def get_pretrained(version="human_v1"):
    """
    Get pre-trained model object.

    Currently Available Models
    ==========================

    See the `documentation <https://d-script.readthedocs.io/en/main/data.html#trained-models>`_ for most up-to-date list.

    - ``lm_v1`` - Language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.
    - ``human_v1`` - Human trained model from D-SCRIPT manuscript.

    Default: ``human_v1``

    :param version: Version of pre-trained model to get
    :type version: str
    :return: Pre-trained model
    :rtype: dscript.models.*
    """
    if not version in VALID_MODELS:
        raise ValueError("Model {} does not exist".format(version))

    state_dict_path = get_state_dict(version)
    return VALID_MODELS[version](state_dict_path)
