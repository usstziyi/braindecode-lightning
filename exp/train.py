"""
模型训练入口：加载指定模型（默认 EEGNet）并训练。

运行方式（在 exp 目录下）：
    uv run python train.py [--model EEGNet]
"""

import argparse
import glob

import torch
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from torchinfo import summary
import torchinfo.layer_info as li

from models import MODEL_REGISTRY


# 利用 Tensor Core 加速矩阵运算（medium/high 会牺牲少量精度换取性能）
torch.set_float32_matmul_precision("high")


def get_layer_name(self, show_var_name, show_depth):
    """层名列显示 模块完整路径 (类名)，如 conv_block.0.conv1 (Conv2d)；根模型无路径，只显示类名"""
    # 沿 parent_info 向上拼出模块在模型内的完整路径；根模型自身不进入路径，
    # LightningModule 的 self.model 只是模型包装属性，同样跳过（避免路径前缀 model.）
    path = ""
    parent = self.parent_info
    while parent is not None and parent.parent_info is not None:
        if parent.var_name == "model":
            break
        path = f"{parent.var_name}.{path}" if path else parent.var_name
        parent = parent.parent_info
    if self.parent_info is not None and self.var_name and self.var_name != "model":
        path = f"{path}.{self.var_name}" if path else self.var_name
    layer_name = f"{path} ({self.class_name})" if path else self.class_name
    if show_depth and self.depth > 0:
        layer_name += f": {self.depth}"
        if self.depth_index is not None:
            layer_name += f"-{self.depth_index}"
    return layer_name


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

    # 让 torchinfo 的 Layer 列显示 模块完整路径 (类名)，如 conv_block.0.conv1 (Conv2d)
    li.LayerInfo.get_layer_name = get_layer_name

    summary(
        lm,
        input_size=lm.model.input_shape,
        col_names=["input_size", "kernel_size", "output_size", "num_params", "trainable"],
        verbose=1,
    )

    checkpoint = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=CONFIG["patience"])
    

    trainer = L.Trainer(
        max_epochs=CONFIG["n_epochs"],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        precision=local_precision(),
        callbacks=[RichProgressBar(leave=True), checkpoint, early_stopping],
    )
    trainer.fit(model=lm, datamodule=dm)

    # trainer.test(model=lm, datamodule=dm) # 直接用刚训练好的模型测试
    trainer.test(model=lm, datamodule=dm, ckpt_path=checkpoint.best_model_path)




if __name__ == "__main__":
    main()
