from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError

from lit_data import LitDataModule


class LitModel(pl.LightningModule):
    """Template Lightning Module to train model"""

    def __init__(self, model_class, lr=0.002, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_class(**kwargs)
        self.lr = lr
        self.sparse = getattr(self.model, "sparse", False)
        self.train_rmse = MeanSquaredError(False)
        self.valid_rmse = MeanSquaredError(False)

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), self.lr)
        else:
            return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def get_loss(self, m_outputs, batch):
        raise NotImplementedError()

    def update_metric(self, m_outputs, batch, partition):
        raise NotImplementedError()

    def forward(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        m_outputs = self(batch)
        loss = self.get_loss(m_outputs, batch)
        self.log("train/loss", loss, sync_dist=True)
        self.update_metric(m_outputs, batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        m_outputs = self(batch)
        loss = self.get_loss(m_outputs, batch)
        self.log("val/loss", loss, sync_dist=True)
        self.update_metric(m_outputs, batch, 'valid')
        return loss

    def training_epoch_end(self, outputs):
        self.log("train/rsme", self.train_rmse.compute(), on_epoch=True, sync_dist=True)
        self.train_rmse.reset()

    def validation_epoch_end(self, outputs):
        self.log("val/rsme", self.valid_rmse.compute(), on_epoch=True, sync_dist=True)
        self.valid_rmse.reset()
