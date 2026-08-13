"""
实战项目训练入口：CIFAR-10 完整训练流程
========================================
整合 Trainer、ModelCheckpoint、EarlyStopping、TensorBoardLogger、
DataModule 与模型，跑通一个完整的图像分类实战。

运行方式：
  uv run python 09_project/train.py
"""

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger

from cifar10_data import CIFAR10DataModule
from model import CIFAR10Model
import torch
import os

# 利用 Tensor Core 加速矩阵运算（medium/high 会牺牲少量精度换取性能）
# 一个是 数学库层 （matmul 用什么精度算）
torch.set_float32_matmul_precision("high")


def main():
    dm = CIFAR10DataModule(
        data_dir="./data",
        batch_size=128,
        num_workers=min(os.cpu_count(), 4),
        val_split=0.1,
    )

    model = CIFAR10Model(num_classes=10, lr=1e-3, t_max=10)

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="cifar10-{epoch}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    logger = TensorBoardLogger(save_dir="logs", name="cifar10")

    # Rich 进度条：保留每个 epoch、彩色、train/val/test 分组显示
    progress_bar = RichProgressBar(
        leave=True,  # 完成后保留进度条（不覆盖）
        # theme=RichProgressBarTheme(
        #     description="green_yellow",      # 进度条左侧描述文本的主题样式
        #     progress_bar="green1",          # 进度条本体的主题样式
        #     progress_bar_finished="green1",  # 已完成进度部分的主题样式
        #     progress_bar_pulse="magenta",   # 进度条脉冲动画的主题样式
        #     batch_progress="grey70",        # batch进度（当前/总批次）的主题样式
        #     time="grey82",                  # 耗时（已用/剩余时间）的主题样式
        #     processing_speed="grey70",      # 处理速度（样本数/秒）的主题样式
        #     metrics="white",                # 指标（如val_acc、val_loss）的主题样式
        # ),
    )

    # 自动选择精度（RTX 4090 原生支持 bf16，优先使用 bf16-mixed）
    # 一个是 训练框架层 （整个 step 怎么分配精度）。
    if torch.cuda.is_available():
        precision = "bf16-mixed"
    elif torch.backends.mps.is_available():
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint, early_stop, progress_bar],
        logger=logger,
        log_every_n_steps=20,
        precision=precision,
    )

    # 训练 + 验证
    trainer.fit(model, datamodule=dm)

    # 在最佳 checkpoint 上测试
    if checkpoint.best_model_path:
        best_model = CIFAR10Model.load_from_checkpoint(checkpoint.best_model_path)
    else:
        best_model = model
    trainer.test(best_model, datamodule=dm)

    print(f"\n最佳模型已保存至: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()