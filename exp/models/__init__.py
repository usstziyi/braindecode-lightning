"""
模型包：模型注册表，统一暴露各模型的公共接口。

用法：
    from exp.models import MODEL_REGISTRY
    m = MODEL_REGISTRY["EEGNet"]
    m.CONFIG, m.build_model, m.load_dataset

新增模型：
    1. 新建 models/<模型类名>/（如 ShallowFBCSPNet/），实现 CONFIG / build_model / load_dataset
    2. 在 MODEL_REGISTRY 中注册
"""

from . import EEGNet
from . import ShallowFBCSPNet
from . import Deep4Net
from . import EEGConformer

MODEL_REGISTRY = {
    "EEGNet": EEGNet,
    "ShallowFBCSPNet": ShallowFBCSPNet,
    "Deep4Net": Deep4Net,
    "EEGConformer": EEGConformer,
}
