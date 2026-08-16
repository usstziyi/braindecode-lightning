import torch.nn as nn



CONFIG = {
    # ---- 数据相关 ----
    "DATASET_NAME": "BNCI2014_001",  # MOABB 数据集名称
    "n_classes": 4,               # 分类数 -> EEGConformer n_outputs
    "n_channels": 22,             # 电极数 -> EEGConformer n_chans


    # ---- 预处理相关 ----
    "bandpass": [4.0, 38.0],     # 通带范围 (Hz)
    "sfreq": 128,                 # 采样率 (Hz)
    "n_times": 512,               # 单段时长(采样点) -> EEGConformer n_times
    "input_window_seconds": 4.0,  # 窗口时长 = n_times / sfreq

    # ---- 模型相关 (对齐 EEGConformer.__init__ 参数) ----
    "n_filters_time": 40,         # 时间卷积核数
    "filter_time_length": 25,     # 时间卷积核长度
    "pool_time_length": 75,       # 时间池化核长度
    "pool_time_stride": 15,       # 时间池化步长
    "drop_prob": 0.5,             # 整体 dropout 概率
    "num_layers": 6,              # Transformer 编码器层数
    "num_heads": 10,              # 多头自注意力头数
    "att_drop_prob": 0.5,         # 注意力 dropout 概率
    "final_fc_length": "auto",    # 自动根据 n_times 计算
    "return_features": False,     # False=输出分类 logits; True=作为特征编码器(如 EEG2Text 前端)
    "activation": nn.ELU,          # 卷积部分激活
    "activation_transfor": nn.GELU,  # Transformer 部分激活

    # ---- 训练相关 ----
    "batch_size": 32,
    "num_workers": 0,
    "n_epochs": 500,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 15,               # early stopping
    "n_folds": 9,                 # 交叉验证折数
    "seed": 42,
}
