"""EEGNet 模型构建。"""

from __future__ import annotations

import torch.nn as nn
from braindecode.models import EEGNet


def build_model(
    n_channels: int,
    n_times: int,
    n_classes: int,
    F1: int = 8,
    D: int = 2,
    F2: int | None = None,  # 默认取 F1 * D
    drop_prob: float = 0.25,
) -> nn.Module:
    """构建 braindecode 的 EEGNet 模型（braindecode >= 1.7 参数命名）。"""
    return EEGNet(
        n_chans=n_channels,
        n_times=n_times,
        n_outputs=n_classes,
        final_conv_length="auto",
        F1=F1,
        D=D,
        F2=F2,
        drop_prob=drop_prob,
    )