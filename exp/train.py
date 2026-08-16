"""
模型训练入口：受试者级 Group K-Fold 交叉验证。

协议：
    对全部受试者做 K 折 GroupKFold（group = 受试者 ID，K = CONFIG["n_folds"] = 9，
    9 个受试者/9 折等价于留一受试者）。
    KFold 循环保留在本脚本；每个 fold 的具体数据准备（加载窗口、按受试者拆分、
    训练折内随机 80/20 验证集）由 DataModule.setup() 完成：
    8 个受试者的全部窗口随机 80/20 划分训练/验证集（early stopping），
    剩余 1 个受试者的全部 session 作为测试集。最终报告各折测试准确率的平均 ± 标准差。

运行方式（在 exp 目录下）：
    uv run python train.py [--model EEGNet]
"""

import argparse

import numpy as np

import torch
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from torchinfo import summary
import torchinfo.layer_info as li

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


def main():
    parser = argparse.ArgumentParser(description="模型训练入口（受试者级 Group K-Fold 交叉验证）")
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

    # 打印一次模型结构
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

    n_folds = CONFIG["n_folds"]
    print(f"受试者级 GroupKFold 折数: {n_folds}")

    fold_accs = []
    for fold in range(n_folds):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        # 数据准备在 DataModule.setup() 内完成（受试者级 GroupKFold + 随机 80/20 验证集）
        dm = m.DataModule(fold=fold, n_folds=n_folds, seed=CONFIG["seed"])

        checkpoint = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            dirpath=f"checkpoints/{args.model}/fold_{fold + 1}",
        )
        early_stopping = EarlyStopping(
            monitor="val_acc", mode="max", patience=CONFIG["patience"]
        )

        trainer = L.Trainer(
            max_epochs=CONFIG["n_epochs"],
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            precision=local_precision(),
            callbacks=[RichProgressBar(leave=True), checkpoint, early_stopping],
        )

        lm = m.LightningModule()
        trainer.fit(model=lm, datamodule=dm)
        results = trainer.test(model=lm, datamodule=dm, ckpt_path=checkpoint.best_model_path)
        acc = results[0]["test_acc"]
        fold_accs.append(acc)
        print(f"Fold {fold + 1} 测试准确率: {acc:.4f}")

    accs = np.array(fold_accs)
    print("\n========== Group K-Fold 交叉验证结果 ==========")
    print(f"每折准确率: {[f'{a:.4f}' for a in accs]}")
    print(f"平均准确率: {accs.mean():.4f} ± {accs.std():.4f}")


if __name__ == "__main__":
    main()
