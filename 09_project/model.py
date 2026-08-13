"""
实战项目模型：CIFAR-10 卷积分类器 LightningModule
==================================================
使用一个轻量 CNN（ResNet 风格残差块），完整演示训练/验证/测试/
预测钩子、torchmetrics 指标、学习率调度与超参保存。
"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy


class ResidualBlock(nn.Module):
    """一个简单的残差块。"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # 升维/降采样 ，扩大通道容量
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(out_ch)
        # 加深变换 ，提取更高阶特征，保持形状以适配残差相加
        # conv1 （3×3）后再接一个 conv2 （3×3），等效感受野从 3×3 扩大到 5×5 ，能捕捉更大范围的局部模式。
        # 这是因为 感受野是"累积"的 ：每个输出像素能看到输入上的一个区域，两个卷积串联后，这个区域会逐层扩大。
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        # 当下采样（ stride≠1 ）或 通道数变化 （ in_ch≠out_ch ）时，才需要真正的 shortcut 卷积。
        # 因为此时主路径的输出尺寸/通道数与输入 x 不一致， 没法直接相加 ，必须用卷积把 x 对齐。
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    # 输入 x
    # ├── 主路径 (main path)：conv1 → bn1 → relu → conv2 → bn2
    # │       ↕
    # │   残差相加：out = 主路径结果 + shortcut(x)
    # │       ↕
    # └── 输出：relu(out)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))    # 1. 卷积→BN→ReLU
        out = self.bn2(self.conv2(out))          # 2. 卷积→BN（先不加激活）
        out += self.shortcut(x)                  # 3. 残差相加
        out = F.relu(out)                        # 4. 最后再 ReLU
        return out


class CIFAR10Model(L.LightningModule):
    def __init__(self, num_classes: int = 10, lr: float = 1e-3, t_max: int = 10):
        super().__init__()
        self.save_hyperparameters()

        self.example_input_array = torch.zeros(1, 3, 32, 32)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), # BatchNorm 会做 平移（bias）和缩放（scale） ，已经具有偏置的作用。
            nn.ReLU(),
            ResidualBlock(32, 32), # 不改变分辨率
            ResidualBlock(32, 64, stride=2), # 升维+下采样
            ResidualBlock(64, 128, stride=2), # 升维+下采样
            # 从而把整张图的信息汇总成每个通道的一个标量
            nn.AdaptiveAvgPool2d(output_size=1), # 压缩到 1x1，保持通道数不变
        )
        self.classifier = nn.Linear(128, num_classes)

        # 指标：官方推荐的 torchmetrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        feats = self.features(x) # (batch_size, 128, 1, 1)
        feats = feats.flatten(1) # (batch_size, 128)
        return self.classifier(feats) # (batch_size, num_classes)

    def _common_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return logits, y, loss

    def training_step(self, batch, batch_idx):
        logits, y, loss = self._common_step(batch)
        self.train_acc(logits, y) # 更新训练准确率指标 ——把当前 batch 的预测结果喂给 train_acc ，让它内部累积正确数/总数。
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, y, loss = self._common_step(batch)
        self.val_acc(logits, y) # ← 这一步：累积指标
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits, y, loss = self._common_step(batch)
        self.test_acc(logits, y) # ← 这一步：累积指标
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        logits = self(x)
        return logits.argmax(dim=1) # ← 这一步：返回预测类别索引

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.t_max
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }