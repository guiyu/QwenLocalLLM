# 文件路径: scripts/train_model.py
# 新建文件

import sys
from pathlib import Path
import logging
import torch
from tqdm import tqdm

# 添加项目根目录到环境变量
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from script.data.dataset_utils import TTSDataset, create_dataloaders
from script.model.tts_model_adapter import TTSModelAdapter, TTSTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_CUDA else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载预训练模型
    logger.info("Loading pretrained model...")
    base_model = AutoModelForCausalLM.from_pretrained(config.OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)
    
    # 创建TTS模型
    logger.info("Creating TTS model...")
    model = TTSModelAdapter(base_model, config)
    
    # 加载数据集
    logger.info("Loading dataset...")
    dataset = TTSDataset(
        data_path=str(Path(config.DATASET_DIR) / "test_dataset.json"),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        split_ratio=config.DATASET_SPLIT_RATIO
    )
    
    # 创建训练器
    trainer = TTSTrainer(model, config, device)
    
    # 开始训练
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for batch in train_pbar:
            loss = trainer.train_step(batch)
            train_losses.append(loss)
            train_pbar.set_postfix({'loss': sum(train_losses)/len(train_losses)})
        
        # 评估阶段
        model.eval()
        val_losses = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
        
        for batch in val_pbar:
            loss = trainer.eval_step(batch)
            val_losses.append(loss)
            val_pbar.set_postfix({'loss': sum(val_losses)/len(val_losses)})
        
        # 计算平均损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # 保存最好的模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = Path(config.OUTPUT_DIR) / "best_tts_model"
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"Saved best model to {model_save_path}")

def main():
    try:
        logger.info("Starting fine-tuning process...")
        train(ModelConfig)
        logger.info("Fine-tuning completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)