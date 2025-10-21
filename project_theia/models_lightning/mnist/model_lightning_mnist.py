from typing import Optional, List
from dataclasses import dataclass, field

import torch
from torch import nn
from lightning import pytorch as pl

from project_theia.models_torch.mnist.mnist_classifier import SimpleMNISTNet, SimpleMNISTNetConfig
from project_theia.training.optimizer import OptimizerConfig, get_lightning_optimizer_dict
from project_theia.data.data_spec import DataSpec


@dataclass
class LitMNISTConfig:
    simple_mnist_net_config: SimpleMNISTNetConfig = field(
        default_factory=SimpleMNISTNetConfig
    )
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    class_weights: Optional[List[float]] = None


class LitMNIST(pl.LightningModule):
    def __init__(self, config: LitMNISTConfig, data_spec: DataSpec, **kwargs):
        super().__init__()
        self.model = SimpleMNISTNet(config=config.simple_mnist_net_config, data_spec=data_spec)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_eval_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_eval_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)