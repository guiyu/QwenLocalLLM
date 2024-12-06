# 文件路径: config/env_config.py
# 新建文件

import os
from pathlib import Path

class EnvConfig:
    # 项目路径
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 环境变量
    ANDROID_HOME = os.getenv('ANDROID_HOME') or os.path.expanduser("~/Android/Sdk")
    ANDROID_NDK_HOME = os.getenv('ANDROID_NDK_HOME') or os.path.join(ANDROID_HOME, "ndk")
    
    # Python环境
    PYTHON_VERSION = "3.9"
    VENV_PATH = PROJECT_ROOT / "venv"
    
    # 模型缓存目录
    HF_CACHE_DIR = os.getenv('HF_CACHE_DIR') or os.path.expanduser("~/.cache/huggingface")
    
    @classmethod
    def validate_environment(cls):
        """验证环境配置"""
        missing = []
        
        if not os.path.exists(cls.ANDROID_HOME):
            missing.append("ANDROID_HOME")
        
        if not os.path.exists(cls.ANDROID_NDK_HOME):
            missing.append("ANDROID_NDK_HOME")
        
        return missing