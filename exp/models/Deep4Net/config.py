import torch.nn as nn



CONFIG = {
    # ---- 数据相关 ----
    "DATASET_NAME": "BNCI2014_001",  # MOABB 数据集名称
    "n_classes": 4,               # 分类数 -> Deep4Net n_outputs
    "n_channels": 22,             # 电极数 -> Deep4Net n_chans


    # ---- 预处理相关 ----
    "bandpass": [4.0, 38.0],     # 通带范围 (Hz)
    "sfreq": 128,                 # 采样率 (Hz)
    "n_times": 512,               # 单段时长(采样点) -> Deep4Net n_times
    "input_window_seconds": 4.0,  # 窗口时长 = n_times / sfreq

    # ---- 模型相关 (对齐 Deep4Net.__init__ 参数) ----
    "final_conv_length": "auto",  # 自动根据 n_times 计算
    "n_filters_time": 25,         # 第一层时间卷积核数
    "n_filters_spat": 25,         # 第一层空间卷积核数
    "filter_time_length": 10,     # 第一层时间卷积核长度
    "pool_time_length": 3,        # 池化核长度
    "pool_time_stride": 3,        # 池化步长
    "n_filters_2": 50,            # 第二层时间卷积核数
    "filter_length_2": 10,        # 第二层时间卷积核长度
    "n_filters_3": 100,           # 第三层时间卷积核数
    "filter_length_3": 10,        # 第三层时间卷积核长度
    "n_filters_4": 200,           # 第四层时间卷积核数
    "filter_length_4": 10,        # 第四层时间卷积核长度
    "activation_first_conv_nonlin": nn.ELU,  # 第一层卷积后激活
    "first_pool_mode": "max",     # 第一层池化方式
    "first_pool_nonlin": nn.Identity,  # 第一层池化后激活
    "activation_later_conv_nonlin": nn.ELU,  # 后续层卷积后激活
    "later_pool_mode": "max",     # 后续层池化方式
    "later_pool_nonlin": nn.Identity,  # 后续层池化后激活
    "drop_prob": 0.5,             # 原 dropout=0.5
    "split_first_layer": True,    # 时间/空间卷积分离成两层
    "batch_norm": True,
    "batch_norm_alpha": 0.1,
    "stride_before_pool": False,  # 决定降采样（时间维缩减）由 卷积的 stride 还是 池化的 stride 来承担。

    # ---- 训练相关 ----
    "batch_size": 32,
    "num_workers": 0,
    "n_epochs": 500,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 15,               # early stopping
    "n_folds": 5,                 # 交叉验证折数
    "seed": 42,
}
