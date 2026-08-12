"""训练超参配置。集中管理数据、模型与训练相关的所有可调参数。"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- 数据 ----
    dataset: str = "BNCI2014_001"  # MOABB 数据集名（BCI Competition IV 2a）
    subject_ids: tuple = (1, 3, 7, 8)  # 参与训练的受试者（BNCI2014_001 只有 1~9）
    bandpass: tuple = (4.0, 38.0)  # 带通滤波范围 (Hz)
    resample_sfreq: float = 128.0  # 重采样频率 (Hz)

    # ---- 建窗（trial 级）----
    trial_start: float = -0.5  # 相对事件起点 (s)
    trial_stop: float = 4.0  # 相对事件终点 (s)

    # ---- 数据划分 ----
    val_ratio: float = 0.2  # 验证集比例

    # ---- 模型 ----
    n_classes: int = 4  # 输出类别数（BCI IV 2a 为 4 类运动想象）

    # ---- 训练 ----
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0
    max_epochs: int = 50
    num_workers: int = 0
    seed: int = 42
    early_stop_patience: int = 10
    log_dir: str = "lightning_logs"