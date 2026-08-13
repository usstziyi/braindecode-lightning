"""
模型包：统一对外暴露各模型的公共接口。

用法：
    from exp.models import CONFIG, build_model, load_dataset
"""

from .eegnet import CONFIG, build_model, load_dataset
