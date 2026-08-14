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
    """显示完整权重形状（含通道维度），格式为 [out, in, kH, kW]；卷积层附加 groups"""
    kernel_size = None
    if hasattr(module, "weight"):
        # 参数化模块（ParametrizationList）没有 shape，需容错
        shape = getattr(module.weight, "shape", None)
        if shape is not None:
            kernel_size = list(shape)
    if kernel_size is None and hasattr(module, "kernel_size"):
        k = module.kernel_size
        kernel_size = list(k) if isinstance(k, (tuple, list)) else int(k)
    if kernel_size is None:
        return None
    groups = getattr(module, "groups", None)
    if groups is not None:
        return f"{kernel_size} (g={groups})"
    return kernel_size


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

    # 让 torchinfo 的 Kernel Shape 列显示完整权重形状（含通道维度）
    li.LayerInfo.get_kernel_size = staticmethod(get_kernel_size)

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
