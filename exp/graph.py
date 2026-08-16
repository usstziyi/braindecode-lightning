"""
模型结构可视化入口：输出 torchinfo 结构摘要 + torchview 结构图。

运行方式（在 exp 目录下）：
    uv run python graph.py [--model EEGNet] [--output-dir graphs] [--filename eegnet_graph]
"""

import argparse
import os

import torch
from torchinfo import summary
import torchinfo.layer_info as li
from torchview import draw_graph

from models import MODEL_REGISTRY


def get_kernel_size(module):
    """卷积层显示 (in,out),(kH,kW),(padding,stride),(groups)；CombinedConv 显示两个子卷积；池化层显示 (kH,kW),(padding,stride)；其余返回 None（显示 --）"""
    def _t(v):
        """把 int/tuple 归一化为 tuple，None 原样返回"""
        if v is None:
            return None
        return tuple(v) if isinstance(v, (tuple, list)) else (v,)

    def _conv_str(conv):
        """单个卷积层的格式串 (in,out),(kH,kW),(padding,stride),(groups)"""
        # 权重形状 [out, in, kH, kW]；参数化模块的 weight 可能没有 shape，退化用 kernel_size
        shape = getattr(conv.weight, "shape", None)
        if shape is not None and len(shape) in (3, 4):
            in_out = (shape[1], shape[0])   # (in, out)
            kernel = tuple(shape[2:])       # (kH, kW) 或 (k,)
        else:
            in_out = (conv.in_channels, conv.out_channels)
            k = conv.kernel_size
            kernel = tuple(k) if isinstance(k, (tuple, list)) else (k,)
        return f"{in_out},{kernel},({_t(conv.padding)},{_t(conv.stride)}),({conv.groups})"

    # braindecode CombinedConv：时间+空间两个卷积融合执行（forward 中不再单独调用子卷积，
    # torchinfo 不会展开它们），这里把两个子卷积都显示出来
    if type(module).__name__ == "CombinedConv" and hasattr(module, "conv_time") and hasattr(module, "conv_spat"):
        return f"{_conv_str(module.conv_time)} ; {_conv_str(module.conv_spat)}"

    # 卷积层（ConvNd 及其子类，含 braindecode 参数化卷积）
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        return _conv_str(module)

    # 池化层（MaxPool/AvgPool）：保留池化窗口信息
    if isinstance(
        module,
        (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
         torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d),
    ):
        return f"{_t(module.kernel_size)},({_t(module.padding)},{_t(module.stride)})"

    return None


def get_layer_name(self, show_var_name, show_depth):
    """层名列显示 模块完整路径 (类名)，如 conv_block.0.conv1 (Conv2d)；根模型无路径，只显示类名"""
    # 沿 parent_info 向上拼出模块在模型内的完整路径；根模型自身不进入路径，
    # LightningModule 的 self.model 只是模型包装属性，同样跳过（避免路径前缀 model.）
    path = ""
    parent = self.parent_info
    while parent is not None and parent.parent_info is not None:
        if parent.var_name == "model":
            break
        path = f"{parent.var_name}.{path}" if path else parent.var_name
        parent = parent.parent_info
    if self.parent_info is not None and self.var_name and self.var_name != "model":
        path = f"{path}.{self.var_name}" if path else self.var_name
    layer_name = f"{path} ({self.class_name})" if path else self.class_name
    if show_depth and self.depth > 0:
        layer_name += f": {self.depth}"
        if self.depth_index is not None:
            layer_name += f"-{self.depth_index}"
    return layer_name


class DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """帮助信息中显示默认值；default=None 的参数不显示（其实际默认值在运行时计算）"""

    def _get_help_string(self, action):
        help_str = super()._get_help_string(action)
        if action.default is None:
            help_str = help_str.replace(" (default: %(default)s)", "")
        return help_str


def main():
    parser = argparse.ArgumentParser(
        description="模型结构可视化入口",
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="EEGNet",
        choices=list(MODEL_REGISTRY.keys()),
        help="模型名（见 models.MODEL_REGISTRY）",
    )
    parser.add_argument(
        "--output-dir",
        default="graphs",
        help="结构图输出目录（相对 exp 目录，或绝对路径）",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="结构图文件名（不含扩展名，默认 <模型名小写>_graph）",
    )
    args = parser.parse_args()

    # 输出目录：相对路径基于脚本所在目录（exp/）
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 让 torchinfo 的 Kernel Shape 列显示 (in,out),(kH,kW),(padding,stride),(groups)
    li.LayerInfo.get_kernel_size = staticmethod(get_kernel_size)
    # 让 torchinfo 的 Layer 列显示 模块完整路径 (类名)，如 conv_block.0.conv1 (Conv2d)
    li.LayerInfo.get_layer_name = get_layer_name

    model = MODEL_REGISTRY[args.model].build_model()

    print(f"input_shape: {model.input_shape}") # (1, n_chans, n_times)
    print(f"input_window_seconds: {model.input_window_seconds}")
    # 卷积类模型为 final_conv_length，EEGConformer 等为 final_fc_length
    if hasattr(model, "final_conv_length"):
        print(f"final_conv_length: {model.final_conv_length}")
    if hasattr(model, "final_fc_length"):
        print(f"final_fc_length: {model.final_fc_length}")

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
    filename = args.filename or f"{args.model.lower()}_graph"
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
        filename=filename,
        directory=output_dir,
        format='png',
        cleanup=True
    )


if __name__ == "__main__":
    main()
