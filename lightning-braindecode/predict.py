"""
推理脚本：从 Hugging Face Hub 加载模型，在测试集上预测并评估准确率。
============================================================
运行方式（在 exp 目录下）：
  uv run python predict.py --repo-id <HF 仓库 ID> [--model EEGNet]
"""

import argparse

import torch

from models import MODEL_REGISTRY
from proxy import _configure_network

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model_name: str, repo_id: str, batch_size: int = 32):
    m = MODEL_REGISTRY[model_name]

    # 从 Hugging Face Hub 加载 braindecode 模型（config + 权重）
    _configure_network()
    net = type(m.build_model()).from_pretrained(repo_id) # 默认eval mode
    print(f"已从 Hub 加载模型: {repo_id} ({type(net).__name__}, n_outputs={net.n_outputs})")

    device = get_device()
    net = net.to(device)

    # 准备测试数据
    dm = m.EEGLightningDataModule()
    dm.batch_size = batch_size
    dm.prepare_data()
    dm.setup("test")

    # 手动推理（无需 Lightning Trainer）
    all_preds, all_y = [], []
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            x, y = x.to(device), y.to(device)
            logits = net(x)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_y.append(y.cpu())
    preds = torch.cat(all_preds)
    y_true = torch.cat(all_y)

    acc = (preds == y_true).float().mean().item()
    print(f"测试样本数: {len(y_true)}")
    print(f"整体准确率: {acc:.4f}")

    for c in range(net.n_outputs):
        mask = y_true == c
        if mask.any():
            class_acc = (preds[mask] == c).float().mean().item()
            print(f"  class {c}: acc = {class_acc:.4f} ({mask.sum().item()} 样本)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理：从 Hugging Face Hub 加载模型并评估")
    parser.add_argument(
        "--model",
        default="EEGNet",
        choices=list(MODEL_REGISTRY.keys()),
        help="模型名（见 models.MODEL_REGISTRY）",
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face 模型仓库 ID，如 usst-ziyi/eegnet")
    parser.add_argument("--batch-size", type=int, default=32, help="推理批大小（默认 32）")
    args = parser.parse_args()

    predict(model_name=args.model, repo_id=args.repo_id, batch_size=args.batch_size)
