# 文件路径: scripts/download_model.py

import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)

def download_phi2_model(output_path: Path) -> bool:
    """下载Phi-2模型"""
    try:
        logger.info(f"Downloading Phi-2 model to {output_path}")
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 下载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        
        # 下载模型
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 保存到本地
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        # 验证文件是否存在
        required_files = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        for file in required_files:
            if not (output_path / file).exists():
                raise Exception(f"Required file missing: {file}")
        
        logger.info("Model download completed")
        return True
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        return False