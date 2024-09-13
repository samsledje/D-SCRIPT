import sys
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from dscript.models.embedding import FullyConnectedEmbed
from dscript.models.contact import ContactCNN
from dscript.models.interaction import ModelInteraction, LogisticActivation

class DSCRIPTModel(ModelInteraction, PyTorchModelHubMixin):
    def __init__(
        self,
        embedding,
        contact,
        use_cuda,
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        super(DSCRIPTModel, self).__init__(
            embedding=embedding,
            contact=contact,
            use_cuda=use_cuda,
            do_w=do_w,
            do_sigmoid=do_sigmoid,
            do_pool=do_pool,
            pool_size=pool_size,
            theta_init=theta_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
        )


if __name__ == "__main__":
    model_pt = sys.argv[1]

    # Load the state dict
    state_dict = torch.load(model_pt)

    # Load the model
    embedding_model = FullyConnectedEmbed(
            6165, 100, dropout=0.5
        )
    contact_model = ContactCNN(100, 500, 7)
    model = ModelInteraction(
            embedding_model,
            contact_model,
            False,
            do_w=True,
            pool_size=9,
            do_pool=True,
            do_sigmoid=False,
        )