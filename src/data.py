"""数据加载、预处理、建窗与 DataLoader 构造。

流程：MOABB 加载 -> Preprocessor 预处理 -> create_windows_from_events 建 trial 窗 ->
      train/val 划分 -> 返回 (train_dataloader, val_dataloader)。
"""

from __future__ import annotations

import numpy as np
import torch
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Filter,
    PickTypes,
    Preprocessor,
    Resample,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from config import Config


def load_windows(config: Config):
    """加载数据集并返回 trial 级窗口数据集。"""
    dataset = MOABBDataset(
        dataset_name=config.dataset,
        subject_ids=list(config.subject_ids),
    )

    preprocessors = [
        PickTypes(eeg=True, meg=False, stim=False, verbose=False),
        Preprocessor(lambda x: x / 1e6, apply_on_array=True),
        Filter(l_freq=config.bandpass[0], h_freq=config.bandpass[1], verbose=False),
        Resample(sfreq=config.resample_sfreq, verbose=False),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
            apply_on_array=True,
        ),
    ]
    preprocess(dataset, preprocessors)

    # 秒 -> 采样点（在重采样后的频率下）
    sfreq = config.resample_sfreq
    trial_start_offset_samples = int(config.trial_start * sfreq)
    # 注释时长已覆盖 cue 之后的 trial 时长，窗口终点落在 trial 结束处
    trial_stop_offset_samples = 0
    window_size = int((config.trial_stop - config.trial_start) * sfreq)

    windows = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        window_size_samples=window_size,
        window_stride_samples=window_size,
        preload=True,
    )
    return windows


def _collate_fn(batch):
    """braindecode 窗口数据集返回 (X, y, crop_inds)，需自定义 collate。"""
    xs, ys, ids = zip(*batch)
    # 每个样本 X 形状为 (n_channels, n_times)，沿 batch 维堆叠 -> (B, C, T)
    X = torch.as_tensor(np.stack(xs, axis=0))
    y = torch.as_tensor(ys, dtype=torch.long)
    indices = torch.as_tensor(ids, dtype=torch.long)
    return X, y, indices


def make_dataloaders(windows, config: Config) -> tuple[DataLoader, DataLoader]:
    """按比例划分 train/val 并返回 DataLoader。"""
    n_total = len(windows)
    indices = np.arange(n_total)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.val_ratio,
        random_state=config.seed,
        stratify=windows.get_metadata()["target"].values,
    )

    train_ds = Subset(windows, train_idx)
    val_ds = Subset(windows, val_idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate_fn,
    )
    return train_dl, val_dl


def get_signal_shape(windows) -> tuple[int, int]:
    """返回 (n_channels, n_times)，用于构建模型。"""
    X = windows[0][0]  # (n_channels, n_times)
    return X.shape[0], X.shape[1]