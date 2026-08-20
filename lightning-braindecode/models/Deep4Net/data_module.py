import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import lightning as L

from .config import CONFIG
from .datasets import load_dataset


# DataLoader 会自动从 train_windows 逐个取数据并组装成 batch
def custom_collate(batch):
    # batch: list of tuples (x, y, crop_inds)
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    crop_inds = [item[2] for item in batch]
    return (
        torch.tensor(np.stack(xs)),         # X: (batch_size, n_channels, n_times)
        torch.tensor(ys),                   # y: (batch_size,)
        # torch.tensor(np.stack(crop_inds)) # (batch_size, 3)
    )

def custom_collate_super(batch):
    batch = torch.utils.data.default_collate(batch)
    return (
        batch[0],                          # X: (batch_size, n_channels, n_times)
        batch[1],                          # y: (batch_size,)
        # torch.tensor(np.stack(batch[2])) # (batch_size, 3)
    )

class EEGLightningDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = CONFIG["batch_size"]
        self.num_workers = CONFIG["num_workers"]

        # MPS 下多进程 DataLoader，强制单进程
        if torch.backends.mps.is_available():
            self.num_workers = 0

        self.save_hyperparameters()

    # 主显卡执行一次
    def prepare_data(self):
        self.train_dataset, self.val_dataset, self.test_dataset = load_dataset()

    # 所有显卡都执行
    def setup(self, stage=None):
        # train/val/test 已在 prepare_data 中按 subject 拆分完成
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_super,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_super,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_super,
            num_workers=self.num_workers,
        )
