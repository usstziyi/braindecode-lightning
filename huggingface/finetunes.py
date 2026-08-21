import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report

from proxy import _configure_network  # 网络策略：直连优先，失败走代理

class CustomDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 编码文本（返回一维 list，无 batch 维）
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_with_custom_dataset():
    """使用自定义数据集训练"""
    # 配置网络（下载模型/推送 Hub 前调用一次即可，进程内生效）
    _configure_network()

    # 准备数据
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
        "Would not recommend to anyone"
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    
    # 分割数据
    split_idx = int(len(texts) * 0.8)
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # 加载模型和分词器
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 创建数据集
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./huggingface/results", # 模型/日志/检查点保存的根目录
        hub_model_id="usst-ziyi/distilbert-sentiment", # Hugging Face Hub 仓库 ID（训练结束推送目标）
        hub_private_repo=True,              # 开启私有仓库模式
        num_train_epochs=10,                # 训练轮数（10 个 epoch）
        per_device_train_batch_size=4,      # 每个设备(train)的批大小
        per_device_eval_batch_size=4,       # 每个设备(eval)的批大小
        learning_rate=2e-5,                 # 学习率
        warmup_steps=1,                     # 预热步数（前 1 步线性升温到 lr）
        logging_strategy="epoch",           # 每个 epoch 结束打印一次日志
        eval_strategy="epoch",              # 评估策略：每个 epoch 结束评估一次
        save_strategy="epoch",              # 每个 epoch 结束保存一次检查点
        save_total_limit=1,                 # 磁盘上只保留 eval_loss 最优的那一个 checkpoint
        load_best_model_at_end=True,        # 训练结束时加载最优检查点
        metric_for_best_model="eval_loss",  # 用于挑选最优检查点的指标
        greater_is_better=False,            # eval_loss 越小越好
        dataloader_pin_memory=False,        # 禁用 pin_memory，避免 MPS 内存问题
        disable_tqdm=True,                  # 关闭 tqdm 进度条，避免与日志 dict 混行
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # 随模型一并上传分词器，保证 Hub 上的模型可直接使用
    )
    
    # 训练
    trainer.train()
    
    return trainer, model, tokenizer

# 运行训练
if __name__ == "__main__":
    trainer, model, tokenizer = train_with_custom_dataset()
    
    # 保存模型
    model.save_pretrained("./huggingface/trained_model")
    tokenizer.save_pretrained("./huggingface/trained_model")

    # model.push_to_hub(
    #     repo_id="usst-ziyi/distilbert-sentiment",
    #     private=True, 
    #     commit_message="Upload trained model",
    # )
    # tokenizer.push_to_hub(
    #     repo_id="usst-ziyi/distilbert-sentiment",
    #     private=True,
    #     commit_message="Upload tokenizer",
    # )

    # 将最优模型（load_best_model_at_end 已载入内存）推送到 Hugging Face Hub
    trainer.push_to_hub(commit_message="End of training: push best model")

    
    # 从 Hugging Face Hub 下载刚推送的模型/分词器，验证 Hub 上的模型可正常使用
    hub_model = AutoModelForSequenceClassification.from_pretrained("usst-ziyi/distilbert-sentiment")
    hub_tokenizer = AutoTokenizer.from_pretrained("usst-ziyi/distilbert-sentiment")

    # 进行预测
    def predict_sentiment(text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        # 将输入移到模型所在设备，避免 CPU/MPS 设备不匹配
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        return "Positive" if pred == 1 else "Negative"
    
    # 测试
    test_sentences = [
        "This is fantastic, I love it!",
        "This is terrible, I hate it!"
    ]
    
    for sentence in test_sentences:
        result = predict_sentiment(sentence, hub_model, hub_tokenizer)
        print(f"'{sentence}' -> {result}")