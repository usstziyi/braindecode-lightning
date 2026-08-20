"""
实战项目推理脚本：加载训练好的 checkpoint 对单张图片进行分类
=============================================================
运行方式：
  uv run python 09_project/predict.py --ckpt checkpoints/cifar10-xxx.ckpt
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import CIFAR10Model

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)


def main():
    # 创建参数解析器并添加参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint 路径")
    parser.add_argument("--image", required=True, help="待分类图片路径")
    # 解析命令行输入，得到命名空间对象 args ，之后通过 args.ckpt 、 args.image 访问值
    args = parser.parse_args()

    # 从 checkpoint 恢复模型（超参已保存）
    model = CIFAR10Model.load_from_checkpoint(args.ckpt)

    img = Image.open(args.image).convert("RGB")
    # 输入张量要与模型在同一设备（模型可能在 MPS/GPU 上）
    tensor = TEST_TRANSFORM(img).unsqueeze(0).to(model.device)  # (1, 3, 32, 32)

    # model.predict() 内部自动切 eval 模式 + no_grad，等价于 model.eval()+no_grad
    logits = model.predict(tensor)
    pred = logits.argmax(dim=1).item() # 取概率最大的类别索引

    print(f"预测类别: {CIFAR10_CLASSES[pred]} (index={pred})")


if __name__ == "__main__":
    main()

# uv run python 09_project/predict.py --ckpt checkpoints/cifar10-epoch=11-val_acc=0.827.ckpt --image 09_project/image/dog.png