"""EEGNet + Lightning 训练入口。

运行方式：
    uv run python src/train.py
"""

from __future__ import annotations

import random

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from config import Config
from data import load_windows, make_dataloaders, get_signal_shape
from model import build_model
from lightning_module import EEGNetLightningModule


def set_seed(seed: int) -> None:
    """固定随机种子以保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    config = Config()
    set_seed(config.seed)

    # 1. 数据
    print(f"加载数据集 {config.dataset}（受试者 {config.subject_ids}）...")
    windows = load_windows(config)
    n_channels, n_times = get_signal_shape(windows)
    print(f"信号形状: {n_channels} 通道 x {n_times} 采样点, 共 {len(windows)} 个窗口")

    train_dl, val_dl = make_dataloaders(windows, config)

    # 2. 模型
    model = build_model(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=config.n_classes,
    )
    lit_module = EEGNetLightningModule(
        model=model,
        n_classes=config.n_classes,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # 3. 回调与日志
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stop_patience,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath="checkpoints",
            filename="best-{epoch:02d}-{val_acc:.3f}",
        ),
    ]
    logger = CSVLogger(save_dir=config.log_dir, name="eegnet")

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",  # 自动检测 GPU/MPS/CPU
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    # 4. 训练
    trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # 5. 验证集最终评估
    trainer.test(lit_module, dataloaders=val_dl, ckpt_path="best")


if __name__ == "__main__":
    main()