# 文件路径: script/data/dataset_utils.py
# 新建文件

import json
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm

class TTSDatasetBuilder:
    def __init__(self, output_dir="./data/tts_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_test_data(self, num_samples=100):
        """生成测试用的TTS数据集"""
        dataset = []
        
        # 金刚经释义文本列表
        texts = [
            "金刚经开篇说：如是我闻，一时佛在舍卫国祇树给孤独园，与大比丘众千二百五十人俱。这讲述了说法的时间、地点和听法大众。",
            "祇树给孤独园是祇陀太子和给孤独长者共同布施的园林，象征智慧和慈悲的结合。",
            "尔时世尊食时，著衣持钵，入舍卫大城乞食。这表现了佛陀的平常心，即使是大觉者，也要随顺世间法。",
            "于其城中，次第乞已，还至本处。饭食讫，收衣钵，洗足已，敷座而坐。这描述了佛陀日常生活的庄严与如法。",
            "善现是在场比丘中解空第一者，他向佛请法，为后世众生请问修行之法。",
            "佛告诉善现，菩萨应该这样降伏其心：所有众生之类，若卵生、若胎生、若湿生、若化生，我皆令入无余涅槃而灭度之。这是教导修行人要以度众为怀。",
            "应无所住而生其心，这是金刚经的核心教义，教导我们不执著于任何事物而自然生起清净心。",
            "一切有为法，如梦幻泡影，如露亦如电，应作如是观。这形象地说明了世间万法的无常本质。",
            "若以色见我，以音声求我，是人行邪道，不能见如来。这告诉我们不要执著于外相。",
            "凡所有相，皆是虚妄。若见诸相非相，即见如来。这教导我们要透过现象看到本质。"
        ]
        
        for i in tqdm(range(num_samples)):
            # 随机选择一个文本
            text = np.random.choice(texts)
            
            # 生成模拟的语音特征
            # 这里我们用随机数模拟mel频谱图
            # 根据文本长度生成相应长度的频谱图
            text_len = len(text)
            time_steps = int(text_len * 5)  # 假设每个字符大约需要5个时间步
            mel_spec = np.random.randn(80, time_steps)
            
            sample = {
                "id": f"sample_{i}",
                "text": text,
                "mel_spectrogram": mel_spec.tolist(),
                "duration": time_steps * 0.0125  # 假设每个时间步是12.5ms
            }
            dataset.append(sample)
        
        # 保存数据集
        with open(self.output_dir / "test_dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        print(f"Generated {num_samples} test samples at {self.output_dir}")
        return str(self.output_dir / "test_dataset.json")

class TTSDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理输入文本
        text_inputs = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 处理目标mel频谱图
        mel_spec = torch.tensor(item["mel_spectrogram"])
        
        return {
            "input_ids": text_inputs["input_ids"].squeeze(),
            "attention_mask": text_inputs["attention_mask"].squeeze(),
            "mel_spectrogram": mel_spec,
            "text": item["text"]
        }

def create_dataloaders(dataset, batch_size=4, split_ratio=0.9):
    """创建训练集和验证集的数据加载器"""
    from torch.utils.data import DataLoader, random_split
    
    # 分割数据集
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader