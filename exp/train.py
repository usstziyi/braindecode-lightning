"""
模型训练入口：加载指定模型（默认 EEGNet）并训练。

运行方式（在 exp 目录下）：
    uv run python train.py [--model EEGNet]
"""

import argparse

import torch
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar
from torchinfo import summary

from models import MODEL_REGISTRY


# 利用 Tensor Core 加速矩阵运算（medium/high 会牺牲少量精度换取性能）
torch.set_float32_matmul_precision("high")


# 自动选择精度（RTX 4090 原生支持 bf16，优先使用 bf16-mixed）
def local_precision():
    if torch.cuda.is_available():
        return "bf16-mixed"
    elif torch.backends.mps.is_available():
        return "bf16-mixed"
    else:
        return "32-true"


def debug_data_module(dm):
    dm.prepare_data()
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    print(f"训练集批次数: {len(train_dataloader)}")
    print(f"验证集批次数: {len(val_dataloader)}")

    dm.setup("test")
    test_dataloader = dm.test_dataloader()
    print(f"测试集批次数: {len(test_dataloader)}")

    for batch in train_dataloader:
        x, y = batch
        print(f"训练集样本形状: {x.shape}")
        print(f"训练集标签形状: {y.shape}")
        break


def main():
    parser = argparse.ArgumentParser(description="模型训练入口")
    parser.add_argument(
        "--model",
        default="EEGNet",
        choices=list(MODEL_REGISTRY.keys()),
        help="模型名（见 models.MODEL_REGISTRY）",
    )
    args = parser.parse_args()

    m = MODEL_REGISTRY[args.model]
    CONFIG = m.CONFIG

    L.seed_everything(CONFIG["seed"])
    dm = m.DataModule()
    lm = m.LightningModule()
    lm.example_input_array = torch.zeros(lm.model.input_shape)

    summary(
        lm,
        input_size=lm.model.input_shape,
        col_names=["input_size", "kernel_size", "output_size", "num_params", "trainable"],
        verbose=1,
    )

    trainer = L.Trainer(
        max_epochs=CONFIG["n_epochs"],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        precision=local_precision(),
        callbacks=[RichProgressBar(leave=True)],
    )
    trainer.fit(model=lm, datamodule=dm)


if __name__ == "__main__":
    main()
