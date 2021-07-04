import torch
import pytorch_lightning as pl
from .embedding import FullyConnectedEmbed
from .contact import ContactCNN
from .interaction import ModelInteraction


class LitInteraction(pl.LightningModule):
    def __init__(self, config):
        super(LitInteraction, self).__init__()

        self.save_hyperparameters(config)

        self._embedding = FullyConnectedEmbed(**config["embed"])
        self._contact = ContactCNN(**config["contact"])
        self._model = ModelInteraction(
            self._embedding, self._contact, **config["interaction"]
        )

    def forward(self, z0, z1):
        return self._model(z0, z1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
