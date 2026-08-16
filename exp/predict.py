"""
推理脚本：加载训练好的 checkpoint，在指定 fold 的测试受试者（全部 session）上预测并评估准确率。
============================================================
运行方式（在 exp 目录下）：
  uv run python predict.py --ckpt <checkpoint 路径> --fold <1~n_folds> [--model EEGNet]
"""

import argparse

import lightning as L
import torch

from models import MODEL_REGISTRY


def predict(model_name: str, ckpt: str, fold: int, batch_size: int = 32):
    m = MODEL_REGISTRY[model_name]
    CONFIG = m.CONFIG

    # 从 checkpoint 恢复模型（权重 + 超参）
    model = m.LightningModule.load_from_checkpoint(ckpt)

    # 与 train.py 同协议：DataModule.setup() 内按 fold 准备数据（受试者级 GroupKFold 拆分）
    dm = m.DataModule(fold=fold, n_folds=CONFIG["n_folds"], seed=CONFIG["seed"])
    dm.batch_size = batch_size
    dm.setup("test")

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
    )

    # predict_step 返回每批预测的类别索引 (batch_size,)
    predictions = trainer.predict(model, datamodule=dm)
    preds = torch.cat(predictions).cpu()  # (n_samples,)

    # 收集真实标签用于评估
    y_true = torch.cat([y for _, y in dm.test_dataloader()]).cpu()  # (n_samples,)

    acc = (preds == y_true).float().mean().item()
    print(f"测试样本数: {len(y_true)}")
    print(f"整体准确率: {acc:.4f}")

    for c in range(model.n_classes):
        mask = y_true == c
        if mask.any():
            class_acc = (preds[mask] == c).float().mean().item()
            print(f"  class {c}: acc = {class_acc:.4f} ({mask.sum().item()} 样本)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理：在指定 fold 的测试受试者上预测并评估")
    parser.add_argument(
        "--model",
        default="EEGNet",
        choices=list(MODEL_REGISTRY.keys()),
        help="模型名（见 models.MODEL_REGISTRY）",
    )
    parser.add_argument("--ckpt", required=True, help="训练好的 checkpoint 路径 (*.ckpt)")
    parser.add_argument("--fold", type=int, required=True, help="第几折（1~n_folds）")
    parser.add_argument("--batch_size", type=int, default=32, help="推理批大小（默认 32）")
    args = parser.parse_args()

    # fold 对外是 1 起，DataModule 内部是 0 起
    predict(model_name=args.model, ckpt=args.ckpt, fold=args.fold - 1, batch_size=args.batch_size)
