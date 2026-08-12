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
from braindecode.models import EEGNet

from torchinfo import summary
from torchview import draw_graph


def main() -> None:
   
    model = EEGNet(
        n_chans=22,
        n_times=int(4.0 * 128.0),
        sfreq=128.0,
        n_outputs=4,
        final_conv_length="auto",
        final_layer_with_constraint=True, # 开启线性层的约束
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
