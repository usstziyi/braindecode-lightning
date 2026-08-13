import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from eegnet_config import EEGNet_CONFIG as Config
from eegnet_model import build_model
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

class EEGNetLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.n_classes = Config["n_classes"]
        self.lr = Config["lr"]
        self.weight_decay = Config["weight_decay"]



        self.save_hyperparameters()

        self.model = build_model()
        
        metric_kwargs = {"task": "multiclass", "num_classes": self.n_classes}
        self.train_metrics = MetricCollection(
            {
                "train_acc": Accuracy(**metric_kwargs),
                # "train_prec": Precision(**metric_kwargs),
                # "train_rec": Recall(**metric_kwargs),
                # "train_f1": F1Score(**metric_kwargs),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "val_acc": Accuracy(**metric_kwargs),
                # "val_prec": Precision(**metric_kwargs),
                # "val_rec": Recall(**metric_kwargs),
                # "val_f1": F1Score(**metric_kwargs),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "test_acc": Accuracy(**metric_kwargs),
                # "test_prec": Precision(**metric_kwargs),
                # "test_rec": Recall(**metric_kwargs),
                # "test_f1": F1Score(**metric_kwargs),
            }
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y= batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_metrics(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_metrics(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics,prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.test_metrics(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log_dict(self.test_metrics,prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer
      
