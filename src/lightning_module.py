"""PyTorch Lightning Module：封装模型的训练/验证/测试循环。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L


class EEGNetLightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_acc(preds, y)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )