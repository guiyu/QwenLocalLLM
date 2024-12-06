# 文件路径: script/model/tts_model_adapter.py
# 新建文件

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class TTSModelAdapter(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.mel_projection = nn.Linear(
            config.hidden_size,
            config.mel_bins
        )
        
    def forward(self, input_ids, attention_mask=None):
        # 获取基础模型的输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 从基础模型输出投影到mel频谱图
        hidden_states = outputs.last_hidden_state
        mel_outputs = self.mel_projection(hidden_states)
        
        return mel_outputs

class TTSTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
    def train_step(self, batch):
        self.model.train()
        
        # 将数据移到指定设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        mel_spectrograms = batch['mel_spectrogram'].to(self.device)
        
        # 前向传播
        outputs = self.model(input_ids, attention_mask)
        
        # 计算损失
        loss = self.criterion(outputs, mel_spectrograms)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            mel_spectrograms = batch['mel_spectrogram'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, mel_spectrograms)
            
        return loss.item()