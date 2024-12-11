# 文件路径: script/download/model_downloader.py

import logging
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shutil
from typing import Tuple, Optional
import hashlib

logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(self.config.HF_CACHE_DIR)
        self.output_dir = Path(self.config.ORIGINAL_MODEL_DIR)
        
        # 确保目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_all_models(self) -> bool:
        """下载所有需要的模型"""
        try:
            logger.info("Starting model download process")
            
            # 1. 下载主对话模型
            if not self.download_llm():
                return False
                
            # 2. 下载ASR模型
            if not self.download_asr():
                return False
                
            # 3. 下载TTS模型
            if not self.download_tts():
                return False
                
            logger.info("All models downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model download failed: {str(e)}")
            return False
    
    def download_llm(self) -> bool:
        """下载Phi-2模型"""
        try:
            logger.info(f"Downloading Phi-2 model from {self.config.BASE_MODEL}")
            
            # 下载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.BASE_MODEL,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # 下载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.config.BASE_MODEL,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 保存到指定目录
            model_path = self.output_dir / "phi-2"
            model.save_pretrained(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # 验证下载
            if not self._verify_model(model_path):
                raise Exception("Model verification failed")
            
            logger.info(f"Phi-2 model downloaded to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Phi-2 model download failed: {str(e)}")
            return False
            
    def download_asr(self) -> bool:
        """下载ASR模型"""
        try:
            logger.info(f"Downloading Whisper model from {self.config.ASR_MODEL}")
            
            # TODO: 实现Whisper.cpp模型下载
            # 这里需要根据具体使用的Whisper.cpp版本实现下载逻辑
            
            return True
            
        except Exception as e:
            logger.error(f"ASR model download failed: {str(e)}")
            return False
            
    def download_tts(self) -> bool:
        """下载TTS模型"""
        try:
            logger.info(f"Downloading FastSpeech2 model from {self.config.TTS_MODEL}")
            
            # TODO: 实现FastSpeech2模型下载
            # 这里需要根据具体使用的FastSpeech2版本实现下载逻辑
            
            return True
            
        except Exception as e:
            logger.error(f"TTS model download failed: {str(e)}")
            return False
            
    def _verify_model(self, model_path: Path) -> bool:
        """验证模型文件完整性"""
        try:
            required_files = [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
            
            # 检查必需文件
            for file in required_files:
                if not (model_path / file).exists():
                    logger.error(f"Missing required file: {file}")
                    return False
            
            # 验证模型加载
            try:
                _ = AutoTokenizer.from_pretrained(str(model_path))
                _ = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16
                )
            except Exception as e:
                logger.error(f"Model loading verification failed: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            return False
    
    def _calculate_md5(self, file_path: Path) -> str:
        """计算文件MD5"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {}
        try:
            model_path = self.output_dir / "phi-2"
            if model_path.exists():
                # 获取模型大小
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                
                # 获取模型文件列表
                files = [f.name for f in model_path.rglob('*') if f.is_file()]
                
                info['model_path'] = str(model_path)
                info['total_size'] = total_size / (1024 * 1024)  # MB
                info['files'] = files
                
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            
        return info