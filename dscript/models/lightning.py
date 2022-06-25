import logging as lg

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .contact import ContactCNN
from .embedding import FullyConnectedEmbed
from .interaction import ModelInteraction

logg = lg.getLogger("D-SCRIPT")


class LitInteraction(pl.LightningModule):
    def __init__(
        self,
        projection_dim: int = 100,
        dropout_p: float = 0.5,
        projection_activation=nn.ReLU,
        hidden_dim: int = 50,
        kernel_width: int = 7,
        contact_activation=nn.Sigmoid,
        pool_width: int = 9,
        lr: float = 1e-3,
        weight_decay: float = 0,
        lambda_similarity: float = 0.35,
        save_prefix: str = "ModelCheckpoint",
        save_every: int = 1,
    ):
        super(LitInteraction, self).__init__()

        self._embedding = FullyConnectedEmbed(
            nin=6165,
            nout=projection_dim,
            dropout=dropout_p,
            activation=projection_activation(),
        )
        self._contact = ContactCNN(
            embed_dim=projection_dim,
            hidden_dim=hidden_dim,
            width=kernel_width,
            activation=contact_activation(),
        )
        self._model = ModelInteraction(
            self._embedding,
            self._contact,
            pool_size=pool_width,
        )
        self.criterion = nn.BCELoss()

        self.train_acc = torchmetrics.Accuracy()
        self.train_aupr = torchmetrics.AveragePrecision()

        self.val_acc = torchmetrics.Accuracy()
        self.val_aupr = torchmetrics.AveragePrecision()

        self.test_acc = torchmetrics.Accuracy()
        self.test_aupr = torchmetrics.AveragePrecision()
        self.test_auroc = torchmetrics.AUROC()
        self.test_spec = torchmetrics.Specificity()
        self.test_sens = torchmetrics.Recall()

        self.save_hyperparameters()
        self.hparams.patience = 3

    def forward(self, z0, z1):
        return self._model(z0, z1)

    def on_train_start(self):
        self.logger.log_hyperparams(params=self.hparams)

    def training_step(self, batch, batch_idx):
        preds = []
        cmaps = []
        labels = []
        for (x0, x1, y) in zip(*batch):
            cmap, y_hat = self._model.map_predict(x0, x1)
            cmaps.append(torch.mean(cmap))
            preds.append(torch.mean(y_hat))
            labels.append(y)

        preds = torch.stack(preds, 0).float()
        cmaps = torch.stack(cmaps, 0)
        labels = torch.stack(labels, 0)

        bce_loss = self.criterion(preds, labels.float())
        cmap_loss = torch.mean(cmaps)
        loss = (self.hparams.lambda_similarity * bce_loss) + (
            (1 - self.hparams.lambda_similarity) * cmap_loss
        )

        self.train_acc(preds.detach(), labels)
        self.train_aupr(preds.detach(), labels)
        train_metrics = {
            "train/bce_loss": bce_loss,
            "train/cmap_loss": cmap_loss,
            "train/loss": loss,
            "train/acc": self.train_acc,
            "train/aupr": self.train_aupr,
        }
        self.log_dict(train_metrics, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = []
        labels = []
        for (x0, x1, y) in zip(*batch):
            y_hat = self._model(x0, x1)
            preds.append(y_hat)
            labels.append(y)
        preds = torch.stack(preds, 0).float()
        labels = torch.stack(labels, 0)

        bce_loss = self.criterion(preds, labels.float())

        self.val_acc(preds.detach(), labels)
        self.val_aupr(preds.detach(), labels)
        val_metrics = {
            "val/bce_loss": bce_loss,
            "val/loss": bce_loss,
            "val/acc": self.val_acc,
            "val/aupr": self.val_aupr,
        }
        self.log_dict(
            val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return bce_loss

    def test_step(self, batch, batch_idx):
        preds = []
        labels = []
        for (x0, x1, y) in zip(*batch):
            y_hat = self._model(x0, x1)
            preds.append(y_hat)
            labels.append(y)
        preds = torch.stack(preds, 0).float()
        labels = torch.stack(labels, 0)

        bce_loss = self.criterion(preds, labels.float())

        self.test_acc(preds, labels)
        self.test_aupr(preds, labels)
        self.test_auroc.update(preds, labels)
        self.test_spec.update(preds, labels)
        self.test_sens.update(preds, labels)
        test_metrics = {
            "test/bce_loss": bce_loss,
            "test/loss": bce_loss,
            "test/acc": self.test_acc,
            "test/aupr": self.test_aupr,
            "test/auroc": self.test_auroc,
            "test/spec": self.test_spec,
            "test/sens": self.test_sens,
        }

        self.log_dict(
            test_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return bce_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return {"optimizer": optimizer}

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val/aupr", mode="max", patience=self.hparams.patience
        )
        checkpoint = ModelCheckpoint(
            monitor="val/aupr",
            dirpath=self.hparams.save_prefix,
            save_top_k=self.hparams.save_every,
            filename="{epoch}-{val/aupr:.2f}",
        )
        return [early_stop, checkpoint]
