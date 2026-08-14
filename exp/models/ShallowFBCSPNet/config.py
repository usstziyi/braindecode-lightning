from braindecode.modules.activation import Square, SafeLog



CONFIG = {
    # ---- 数据相关 ----
    "DATASET_NAME": "BNCI2014_001",  # MOABB 数据集名称
    "n_classes": 4,               # 分类数 -> ShallowFBCSPNet n_outputs
    "n_channels": 22,             # 电极数 -> ShallowFBCSPNet n_chans


    # ---- 预处理相关 ----
    "bandpass": [4.0, 38.0],     # 通带范围 (Hz)
    "sfreq": 128,                 # 采样率 (Hz)
    "n_times": 512,               # 单段时长(采样点) -> ShallowFBCSPNet n_times
    "input_window_seconds": 4.0,  # 窗口时长 = n_times / sfreq

    # ---- 模型相关 (对齐 ShallowFBCSPNet.__init__ 参数) ----
    "n_filters_time": 40,         # 第一层时间卷积核数
    "filter_time_length": 25,     # 时间卷积核长度
    "n_filters_spat": 40,         # 空间卷积核数
    "pool_time_length": 75,       # 时间池化核长度
    "pool_time_stride": 15,       # 时间池化步长
    "final_conv_length": "auto",  # 自动根据 n_times 计算
    "conv_nonlin": Square,        # 平方非线性激活
    "pool_mode": "mean",          # 池化方式
    "activation_pool_nonlin": SafeLog,  # 池化后非线性激活
    "split_first_layer": True,    # 时间/空间卷积分离成两层
    "batch_norm": True,
    "batch_norm_alpha": 0.1,
    "drop_prob": 0.5,             # 原 dropout=0.5

    # ---- 训练相关 ----
    "batch_size": 32,
    "num_workers": 0,
    "n_epochs": 5,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 15,               # early stopping
    "n_folds": 5,                 # 交叉验证折数
    "seed": 42,
}
