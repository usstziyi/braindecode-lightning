"""演示：用 BNCI2014_001 数据集 + mne 正确推导信号维度并初始化 EEGNet。

运行方式（从项目根目录）：
    uv run python viz/demo_init_eegnet.py

要点：
1. 用 MOABBDataset 加载数据（底层就是 mne 对象）。
2. 直接从 mne 的 info 读取采样率 sfreq、通道数 n_chans。
3. 由窗口时长 (trial_stop - trial_start) × sfreq 得到 n_times。
4. 用这些真实维度初始化 EEGNet，并做一次前向传播验证。

本脚本为自包含 demo，直接使用 braindecode 提供的接口，不依赖 src 下的模块。
"""



import torch
from numpy import multiply
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGNet
from braindecode.preprocessing import (
    Filter,
    PickTypes,
    Preprocessor,
    Resample,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from torchinfo import summary
from torchview import draw_graph



# V to µV
def scale_to_microvolt(data):
    return multiply(data, 1e6)

def main() -> None:
    dataset = MOABBDataset(
        dataset_name="BNCI2014_001", 
        subject_ids=[1],
    )

    preprocessors = [
        PickTypes(eeg=True, stim=False, verbose=False),
        Preprocessor(scale_to_microvolt),
        Filter(l_freq=4.0, h_freq=38.0, verbose=False),
        Resample(sfreq=128.0, verbose=False),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
            apply_on_array=True,
        ),
    ]
    preprocess(dataset, preprocessors)

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    n_chans = dataset.datasets[0].raw.info["nchan"]

    print(f"sfreq: {sfreq}")
    print(f"n_chans: {n_chans}")



    model = EEGNet(
        n_chans=n_chans,
        n_times=int(4.0 * sfreq),
        sfreq=sfreq,
        n_outputs=4,
        final_conv_length="auto",
    )

    print(f"input_shape: {model.input_shape}") # (1, n_chans, n_times)
    print(f"input_window_seconds: {model.input_window_seconds}")
    print(f"final_conv_length: {model.final_conv_length}")
    

    # 查看模型结构
    summary(
        model, 
        input_size=model.input_shape,
        col_names=["input_size","kernel_size", "output_size", "num_params","trainable"],
        verbose=1,
    )
    # 查看卷积层的 kernel_size、 stride、 padding、 groups
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            print(name, "kernel_size=", m.kernel_size, "stride=", m.stride, "padding=", m.padding, "groups=", m.groups)

    # 绘制模型结构
    model_graph = draw_graph(
        model, 
        input_size=model.input_shape, 
        device='meta',
        expand_nested=True, # 展开嵌套模块
        hide_module_functions=True, # 隐藏模块函数
        save_graph=False,
        # collect_attributes=True, # 显示模块属性
    )
    # 提高渲染 DPI（默认 96），图片更清晰；也可改成 svg 得到无限清晰矢量图
    model_graph.visual_graph.graph_attr.update({
        'dpi': '300', 
        'splines': 'spline' # spline 曲线更平滑，ortho 直角正交折线, line 直线折线
    })
    model_graph.visual_graph.render(
        filename='eegnet_graph',
        directory='./viz',
        format='png', 
        cleanup=True
    )








if __name__ == "__main__":
    main()
