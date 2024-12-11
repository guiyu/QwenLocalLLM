# 文件路径: config/model_config.py

from pathlib import Path
from .env_config import EnvConfig

class ModelConfig:
    # 项目根路径
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    
    # 模型相关路径
    MODEL_DIR = PROJECT_ROOT / "models"
    ORIGINAL_MODEL_DIR = MODEL_DIR / "original"
    QUANTIZED_MODEL_DIR = MODEL_DIR / "quantized"
    ANDROID_MODEL_DIR = MODEL_DIR / "android"
    
    # 确保目录存在
    ORIGINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    QUANTIZED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ANDROID_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 基础模型配置
    BASE_MODEL = "microsoft/phi-2"
    MODEL_REVISION = "v1.0"
    HF_CACHE_DIR = EnvConfig.HF_CACHE_DIR
    
    # ASR配置
    ASR_MODEL = "alphacep/wav2letter-tiny"
    ASR_VERSION = "v1.0"
    
    # TTS配置
    TTS_MODEL = "coqui/fast-basic"
    TTS_VERSION = "v1.0"
    
    # 量化配置
    QUANTIZATION_CONFIG = {
        'method': 'ggml',
        'bits': 4,
        'groupsize': 32,
        'use_sparse': True
    }
    
    # 内存配置
    MEMORY_CONFIG = {
        'total_limit': 1024 * 1024 * 1024,  # 1GB
        'model_cache': 512 * 1024 * 1024,   # 512MB
        'kv_cache': 256 * 1024 * 1024      # 256MB
    }
    
    # 优化配置
    OPTIMIZATION_CONFIG = {
        'thread_count': 4,
        'batch_size': 8,
        'use_fp16': True
    }
    
    # 输出文件配置
    OUTPUT_CONFIG = {
        'quantized_model_name': 'model_quantized.onnx',
        'model_config_name': 'config.json',
        'vocab_file_name': 'vocab.txt'
    }
    
    # Android端性能目标
    PERFORMANCE_TARGETS = {
        'model_load_time': 3.0,    # 秒
        'first_inference': 0.5,    # 秒
        'inference_time': 0.2,     # 秒
        'memory_usage': 1024,      # MB
        'storage_size': 800        # MB
    }
    
    @classmethod
    def get_model_paths(cls):
        """获取模型文件路径"""
        return {
            'original': cls.ORIGINAL_MODEL_DIR,
            'quantized': cls.QUANTIZED_MODEL_DIR,
            'android': cls.ANDROID_MODEL_DIR,
            'quantized_model': cls.QUANTIZED_MODEL_DIR / cls.OUTPUT_CONFIG['quantized_model_name']
        }
    
    @classmethod
    def get_quantization_config(cls):
        """获取量化配置"""
        return cls.QUANTIZATION_CONFIG
    
    @classmethod
    def get_memory_config(cls):
        """获取内存配置"""
        return cls.MEMORY_CONFIG
    
    @classmethod
    def get_optimization_config(cls):
        """获取优化配置"""
        return cls.OPTIMIZATION_CONFIG
    
    @classmethod
    def validate_android_targets(cls, metrics):
        """验证是否达到Android端性能目标"""
        results = {}
        for key, target in cls.PERFORMANCE_TARGETS.items():
            if key in metrics:
                results[key] = {
                    'target': target,
                    'actual': metrics[key],
                    'achieved': metrics[key] <= target
                }
        return results