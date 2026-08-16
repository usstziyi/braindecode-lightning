import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import lightning as L

from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import GroupKFold

from .config import CONFIG
from .datasets import load_windows_dataset


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

class DataModule(L.LightningDataModule):
    # 全部受试者的窗口数据集只加载一次，各 fold 复用（进程级缓存）
    _windows_cache = None

    def __init__(self, fold, n_folds=None, seed=42):
        super().__init__()
        self.batch_size = CONFIG["batch_size"]
        self.num_workers = CONFIG["num_workers"]

        # MPS 下多进程 DataLoader，强制单进程
        if torch.backends.mps.is_available():
            self.num_workers = 0

        self.fold = fold                    # 交叉验证第几折（0 起）
        self.n_folds = n_folds if n_folds is not None else CONFIG["n_folds"]
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @classmethod
    def _load_windows(cls):
        if cls._windows_cache is None:
            cls._windows_cache = load_windows_dataset()
        return cls._windows_cache

    # 具体数据准备：受试者级 GroupKFold 拆分 + 训练折内随机 80/20 验证集
    def setup(self, stage=None):
        # fit / test 阶段 setup 会被多次调用，已构建则跳过
        if self.train_dataset is not None:
            return

        windows_dataset = self._load_windows()
        subject_splits = windows_dataset.split(by="subject")
        subject_ids = np.array(sorted(subject_splits.keys(), key=str))
        n_subjects = len(subject_ids)
        n_folds = min(self.n_folds, n_subjects)

        # 受试者级 GroupKFold：每个受试者作为一个样本，group 即其自身 ID
        gkf = GroupKFold(n_splits=n_folds)
        folds = list(gkf.split(np.arange(n_subjects)[:, None], groups=subject_ids))
        train_idx, test_idx = folds[self.fold]

        test_subject = subject_ids[test_idx[0]]
        train_subjects = [s for s in subject_ids[train_idx]]
        test_dataset = subject_splits[test_subject]

        # 训练折（其余受试者的全部窗口）随机 80/20 划分出验证集（early stopping）
        train_windows = BaseConcatDataset([subject_splits[s] for s in train_subjects])
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_windows,
            [0.8, 0.2],
            generator=torch.Generator().manual_seed(self.seed + self.fold),
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        print(
            f"[DataModule] Fold {self.fold + 1}/{n_folds} | "
            f"训练: {train_subjects} | 测试: {test_subject} | "
            f"窗口 训练 {len(train_dataset)} / 验证 {len(val_dataset)} / 测试 {len(test_dataset)}"
        )

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
