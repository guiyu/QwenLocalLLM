# 文件路径: script/model/model_adapter.py

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import numpy as np

from utils.exceptions import ModelError

logger = logging.getLogger(__name__)

class BaseModelAdapter:
    """基础模型适配器"""
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载模型"""
        raise NotImplementedError

    def export_weights(self):
        """导出模型权重"""
        raise NotImplementedError

class PhiModelAdapter(BaseModelAdapter):
    """Phi-2模型适配器"""
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "microsoft/phi-2"

    def load_model(self):
        """加载Phi-2模型"""
        try:
            logger.info(f"Loading Phi-2 model from {self.model_name}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.model.eval()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Phi-2 model: {e}")
            raise ModelError(f"Failed to load Phi-2 model: {e}")

    def export_weights(self, output_path: Path):
        """导出权重为GGML格式"""
        try:
            if not self.model:
                raise ModelError("Model not loaded")

            logger.info("Exporting model weights...")
            weights = {}
            
            # 收集所有权重
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    weights[name] = param.detach().cpu().numpy()

            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入二进制文件
            with open(output_path, 'wb') as f:
                # 写入头信息
                header = {
                    'version': 1,
                    'n_weights': len(weights),
                }
                np.save(f, header)
                
                # 写入权重
                for name, weight in weights.items():
                    name_bytes = name.encode('utf-8')
                    f.write(len(name_bytes).to_bytes(4, byteorder='little'))
                    f.write(name_bytes)
                    
                    # 写入权重形状
                    shape = np.array(weight.shape, dtype=np.int32)
                    shape.tofile(f)
                    
                    # 写入权重数据
                    weight.tofile(f)

            logger.info(f"Model weights exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export model weights: {e}")
            raise ModelError(f"Failed to export model weights: {e}")

class WhisperAdapter(BaseModelAdapter):
    """Whisper ASR模型适配器"""
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "alphacep/wav2letter-tiny"

    def load_model(self):
        """加载Whisper模型"""
        try:
            # TODO: 实现Whisper模型加载
            logger.info("WhisperAdapter: load_model not implemented yet")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise ModelError(f"Failed to load Whisper model: {e}")

    def export_weights(self, output_path: Path):
        """导出Whisper权重"""
        try:
            # TODO: 实现Whisper权重导出
            logger.info("WhisperAdapter: export_weights not implemented yet")
            return True
        except Exception as e:
            logger.error(f"Failed to export Whisper weights: {e}")
            raise ModelError(f"Failed to export Whisper weights: {e}")

class FastSpeechAdapter(BaseModelAdapter):
    """FastSpeech TTS模型适配器"""
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "coqui/fast-basic"

    def load_model(self):
        """加载FastSpeech模型"""
        try:
            # TODO: 实现FastSpeech模型加载
            logger.info("FastSpeechAdapter: load_model not implemented yet")
            return True
        except Exception as e:
            logger.error(f"Failed to load FastSpeech model: {e}")
            raise ModelError(f"Failed to load FastSpeech model: {e}")

    def export_weights(self, output_path: Path):
        """导出FastSpeech权重"""
        try:
            # TODO: 实现FastSpeech权重导出
            logger.info("FastSpeechAdapter: export_weights not implemented yet")
            return True
        except Exception as e:
            logger.error(f"Failed to export FastSpeech weights: {e}")
            raise ModelError(f"Failed to export FastSpeech weights: {e}")