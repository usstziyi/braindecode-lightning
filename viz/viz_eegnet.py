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
