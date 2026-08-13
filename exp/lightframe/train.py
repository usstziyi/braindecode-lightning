import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from torchinfo import summary

from ..models.eegnet import CONFIG
from .data_module import EEGNetLightningDataModule
from .module import EEGNetLightningModule



# 利用 Tensor Core 加速矩阵运算（medium/high 会牺牲少量精度换取性能）
# 一个是 数学库层 （matmul 用什么精度算）
torch.set_float32_matmul_precision("high")
# 自动选择精度（RTX 4090 原生支持 bf16，优先使用 bf16-mixed）
# 一个是 训练框架层 （整个 step 怎么分配精度）。
def local_precision():
    if torch.cuda.is_available():
        return "bf16-mixed"
    elif torch.backends.mps.is_available():
        return "bf16-mixed"
    else:
        return "32-true"


def debug_data_module(ldm: EEGNetLightningDataModule):
    ldm.prepare_data()
    ldm.setup("fit")
    train_dataloader = ldm.train_dataloader()
    val_dataloader = ldm.val_dataloader()
    print(f"训练集批次数: {len(train_dataloader)}") 
    print(f"验证集批次数: {len(val_dataloader)}") 

    ldm.setup("test")
    test_dataloader = ldm.test_dataloader()
    print(f"测试集批次数: {len(test_dataloader)}") 

    for batch in train_dataloader:
        x, y = batch
        print(f"训练集样本形状: {x.shape}")
        print(f"训练集标签形状: {y.shape}")
        break



def train():
    L.seed_everything(CONFIG["seed"])
    ldm = EEGNetLightningDataModule()
    lm = EEGNetLightningModule()
    lm.example_input_array = torch.zeros(lm.model.input_shape)

    summary(
        lm, 
        input_size=lm.model.input_shape,
        col_names=["input_size","kernel_size", "output_size", "num_params","trainable"],
        verbose=1,
    )


    trainer = L.Trainer(
        max_epochs=CONFIG["n_epochs"], 
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        precision=local_precision(),
        callbacks=[RichProgressBar(leave=True)],
    )
    trainer.fit(model = lm, datamodule = ldm)


