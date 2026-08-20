"""
实战项目数据模块：CIFAR-10 DataModule
======================================
封装 CIFAR-10 数据集的下载、切分、增强与 DataLoader 生成。
"""

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 2,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # 只执行一次：下载数据集
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # 训练增强：随机裁剪 + 水平翻转 + 归一化
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # 防止过拟合,随机裁剪 32x32，填充 4px
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        # 测试只用归一化
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        full_train = datasets.CIFAR10(
            self.hparams.data_dir, train=True, download=False, transform=train_transform
        )
        if stage in (None, "fit"):
            val_len = int(len(full_train) * self.hparams.val_split)
            train_len = len(full_train) - val_len
            self.train_ds, self.val_ds = random_split(full_train, [train_len, val_len])
        if stage in (None, "test"):
            self.test_ds = datasets.CIFAR10(
                self.hparams.data_dir, train=False, download=False, transform=test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            # 跨 epoch 复用 worker，避免重复初始化（仅在 num_workers > 0 时生效）
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )