import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np


class TTSDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=512):
        self.tokenizer = tokenizer
        self.data = load_dataset(data_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 假设数据集中包含 'text' 和 'speech' 字段
        inputs = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': item['speech']  # 这里需要根据实际数据集格式调整
        }


def fine_tune_model(model, tokenizer, training_args, train_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    print("Starting fine-tuning...")
    trainer.train()

    # 保存微调后的模型
    output_dir = "./fine_tuned_model"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    # 加载之前下载的模型和分词器
    model_path = "./qianwen_model"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_steps=1000,
        logging_steps=100,
        fp16=True  # 使用混合精度训练
    )

    # 准备数据集
    train_dataset = TTSDataset(tokenizer, "path_to_your_tts_dataset")

    # 开始微调
    fine_tune_model(model, tokenizer, training_args, train_dataset)