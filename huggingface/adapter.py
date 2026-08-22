"""
LoRA 高效微调 vs 全量微调：显存占用对比（MPS / macOS）

运行方式（在项目根目录）：
    uv run python huggingface/lora_main.py

说明：
- 同一份小数据集、同样的 TrainingArguments（3 epoch，batch 4）
- 通过 torch.mps.current_allocated_memory() 在每个训练 step 后采样峰值显存
- 全量微调：67.8M 参数全部可训练（梯度 + Adam 状态巨大）
- LoRA：仅 0.887M 参数可训练（基座冻结），显存主要省在梯度与优化器状态
"""

import gc
import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer, TrainingArguments, TrainerCallback,
    AutoTokenizer, AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from proxy import _configure_network  # 直连优先，失败走代理

MODEL_NAME = "distilbert-base-uncased"
HUB_MODEL_ID = "usst-ziyi/distilbert-sentiment-lora"  # 可选：LoRA 推送目标


class CustomDataset(Dataset):
    """自定义数据集类（与 main.py 一致）"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
        )
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------- 数据 ----------------
def make_datasets(tokenizer):
    texts = [
        "The product is excellent and works perfectly",
        "Very satisfied with the quality and service",
        "Amazing experience, will buy again",
        "Great value for money",
        "The best purchase I've made this year",
        "Poor quality, broke after one day",
        "Terrible customer service experience",
        "Not worth the money at all",
        "Very disappointed with this product",
        "Would not recommend to anyone",
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    split = int(len(texts) * 0.8)
    train_ds = CustomDataset(texts[:split], labels[:split], tokenizer)
    val_ds = CustomDataset(texts[split:], labels[split:], tokenizer)
    return train_ds, val_ds


# ---------------- 显存采样 ----------------
class MemoryTracker:
    def __init__(self):
        self.peak_mb = 0.0

    def sample(self):
        """MPS上 采样当前已分配显存（MB），并记录峰值"""
        if torch.backends.mps.is_available():
            mb = torch.mps.current_allocated_memory() / 1024 ** 2
            self.peak_mb = max(self.peak_mb, mb)
        return self.peak_mb


class MemoryCallback(TrainerCallback):
    """每个训练 step 结束后采样一次显存，捕获峰值"""
    def __init__(self, tracker):
        self.tracker = tracker

    def on_step_end(self, args, state, control, **kwargs):
        self.tracker.sample()


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        hub_model_id=HUB_MODEL_ID,       # Hub 仓库 ID（trainer.push_to_hub 推送目标）
        hub_private_repo=True,           # 私有仓库
        num_train_epochs=3,                # 3 个 epoch，够看趋势且跑得快
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        warmup_steps=1,
        logging_strategy="no",             # 对比显存，不打印日志
        eval_strategy="epoch",
        save_strategy="no",                # 不落盘 checkpoint
        dataloader_pin_memory=False,       # 禁用 pin_memory，避免 MPS 内存问题
        disable_tqdm=True,
        report_to=[],                      # 不接 tensorboard/wandb
    )


# ---------------- 1) 全量微调 ----------------
def train_full_finetune(tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    trainable = count_trainable(model)

    train_ds, val_ds = make_datasets(tokenizer)
    tracker = MemoryTracker()
    trainer = Trainer(
        model=model,
        args=build_args("./huggingface/results/results_full"),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[MemoryCallback(tracker)],
    )
    trainer.train()
    peak_mb = tracker.peak_mb

    # 释放，为 LoRA 让出显存
    del trainer, model
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return trainable, peak_mb


# ---------------- 2) LoRA 微调 ----------------
def train_lora(tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,        # 序列分类任务
        r=8,                               # LoRA 秩
        lora_alpha=16,                     # 缩放系数
        lora_dropout=0.1,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # DistilBERT attention 线性层
        bias="none",
    )
    # 使用 LoRA 配置将基座模型包装为 PEFT 模型，仅训练新增的低秩适配层
    model = get_peft_model(model, lora_config)
    trainable = count_trainable(model)

    train_ds, val_ds = make_datasets(tokenizer)
    tracker = MemoryTracker()
    trainer = Trainer(
        model=model,
        args=build_args("./huggingface/results/results_lora"),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[MemoryCallback(tracker)],
    )
    trainer.train()
    peak_mb = tracker.peak_mb

    # 保存 LoRA adapter到本地（很小，只有几 MB）
    model.save_pretrained("./huggingface/results/results_adapter")
    tokenizer.save_pretrained("./huggingface/results/results_adapter")

    # 将 adapter 推送到 Hub（需要 hub_model_id + 已登录 HF）
    trainer.push_to_hub(commit_message="Upload LoRA adapter")

    del trainer
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return model, trainable, peak_mb


# ---------------- 推理验证（加载 adapter） ----------------
def predict_with_lora():
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = PeftModel.from_pretrained(base, HUB_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(HUB_MODEL_ID)

    device = next(model.parameters()).device
    for text in ["This is fantastic, I love it!", "This is terrible, I hate it!"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        print(f"  '{text}' -> {'Positive' if pred == 1 else 'Negative'}")


if __name__ == "__main__":
    _configure_network()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("=" * 56)
    print("1) 全量微调 (Full Fine-tuning)")
    print("=" * 56)
    full_trainable, full_peak = train_full_finetune(tokenizer)
    print(f"   可训练参数: {full_trainable/1e6:8.2f} M | 峰值显存: {full_peak:8.1f} MB")

    print("\n" + "=" * 56)
    print("2) LoRA 高效微调")
    print("=" * 56)
    lora_model, lora_trainable, lora_peak = train_lora(tokenizer)
    print(f"   可训练参数: {lora_trainable/1e6:8.3f} M | 峰值显存: {lora_peak:8.1f} MB")

    print("\n" + "=" * 56)
    print("对比结果")
    print("=" * 56)
    print(f"   可训练参数缩减: {full_trainable / lora_trainable:8.1f} x")
    print(f"   峰值显存缩减:   {full_peak / lora_peak:8.1f} x")

    print("\nLoRA adapter 推理验证:")
    predict_with_lora()
