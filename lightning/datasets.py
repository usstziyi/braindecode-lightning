import torch
from numpy import multiply
from braindecode.datasets import MOABBDataset
from eegnet_model import build_model
from braindecode.preprocessing import (
    Filter,
    PickTypes,
    Resample,
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from eegnet_config import EEGNet_CONFIG as CONFIG

# V to µV
def scale_to_microvolt(data):
    return multiply(data, 1e6)

def load_dataset():
    dataset = MOABBDataset(
        dataset_name=CONFIG["DATASET_NAME"], 
        # subject_ids=[CONFIG["subject_id"]],
    )

    preprocessors = [
        PickTypes(eeg=True, stim=False, verbose=False),
        Preprocessor(scale_to_microvolt),
        Filter(l_freq=CONFIG["bandpass"][0], h_freq=CONFIG["bandpass"][1], verbose=False),
        Resample(sfreq=CONFIG["sfreq"], verbose=False),
        Preprocessor(
            exponential_moving_standardize,  # 指数移动标准化函数，用于对数据进行标准化处理
            factor_new=1e-3,  # 指数移动平均的平滑因子，控制历史数据的衰减速率，值越小越依赖历史
            init_block_size=1000,  # 初始化块大小，用于计算初始统计量（均值和标准差）的样本数量
        ),
    ]
    preprocess(dataset, preprocessors)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=CONFIG["n_times"],
        window_stride_samples=CONFIG["n_times"],
        preload=True,
        verbose=False,
    )

    # split
    splits = windows_dataset.split(by="session")
    train_windows_dataset = splits["0train"]
    test_windows_dataset = splits["1test"]


    # 打印数据集大小
    print(f"训练集大小(窗口数): {len(train_windows_dataset)}")
    print(f"测试集大小(窗口数): {len(test_windows_dataset)}")

    return train_windows_dataset, test_windows_dataset
