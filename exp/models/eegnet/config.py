import torch.nn as nn



CONFIG = {
    # ---- 数据相关 ----
    "DATASET_NAME": "BNCI2014_001",  # MOABB 数据集名称
    "subject_id": 1,              # MOABB 受试者
    "n_classes": 4,               # 分类数 -> EEGNet n_outputs
    "n_channels": 22,             # 电极数 -> EEGNet n_chans


    # ---- 预处理相关 ----
    "bandpass": [4.0, 38.0],     # 通带范围 (Hz)
    "sfreq": 128,                 # 采样率 (Hz)
    "n_times": 512,               # 单段时长(采样点) -> EEGNet n_times
    "input_window_seconds": 4.0,  # 窗口时长 = n_times / sfreq

    # ---- 模型相关 (对齐 EEGNet.__init__ 参数) ----
    "final_conv_length": "auto",  # 自动根据 n_times 计算
    "pool_mode": "mean",
    "F1": 8,                      # 第一层时间卷积核数
    "D": 2,                       # depthwise 深度乘子
    "F2": None,                   # None -> 自动取 F1 * D
    "kernel_length": 64,          # 时间卷积核长度
    "depthwise_kernel_length": 16,
    "pool1_kernel_size": 4,
    "pool2_kernel_size": 8,
    "conv_spatial_max_norm": 1,
    "activation": nn.ELU,          # nn.ELU
    "batch_norm_momentum": 0.01,
    "batch_norm_affine": True,
    "batch_norm_eps": 1e-3,
    "drop_prob": 0.25,            # 原 dropout=0.3, 对齐 EEGNet 参数名
    "final_layer_with_constraint": True,
    "norm_rate": 0.25,

    # ---- 训练相关 ----
    "batch_size": 32,
    "num_workers": 4,
    "n_epochs": 200,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 15,               # early stopping
    "n_folds": 5,                 # 交叉验证折数
    "seed": 42,
}
