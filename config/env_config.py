# 文件路径: config/env_config.py

import os
import platform
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EnvConfig:
    # 项目路径
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    
    # 环境变量
    ANDROID_HOME = os.getenv('ANDROID_HOME') or os.path.expanduser("~/Android/Sdk")
    ANDROID_NDK_HOME = os.getenv('ANDROID_NDK_HOME') or os.path.join(ANDROID_HOME, "ndk")
    
    # Python环境
    PYTHON_VERSION = "3.9"
    VENV_PATH = PROJECT_ROOT / "venv"
    
    # 模型缓存目录
    HF_CACHE_DIR = os.getenv('HF_CACHE_DIR') or os.path.expanduser("~/.cache/huggingface")
    
    # CUDA配置
    CUDA_HOME = os.getenv('CUDA_HOME')
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    
    # MTK设备配置
    MTK_CONFIG = {
        'memory_alignment': 16,    # 字节对齐
        'page_size': 4096,         # 内存页大小
        'chunk_size': 1024 * 1024  # 加载块大小
    }
    
    # 系统信息
    SYSTEM_INFO = {
        'os': platform.system(),
        'arch': platform.machine(),
        'python_version': platform.python_version(),
        'cuda_available': CUDA_HOME is not None
    }
    
    @classmethod
    def validate_environment(cls) -> list:
        """验证环境配置"""
        missing = []
        
        # 检查Android SDK
        if not os.path.exists(cls.ANDROID_HOME):
            missing.append("ANDROID_HOME")
            logger.error(f"Android SDK not found at: {cls.ANDROID_HOME}")
        
        # 检查Android NDK
        if not os.path.exists(cls.ANDROID_NDK_HOME):
            missing.append("ANDROID_NDK_HOME")
            logger.error(f"Android NDK not found at: {cls.ANDROID_NDK_HOME}")
        
        # 检查Python版本
        if platform.python_version() < cls.PYTHON_VERSION:
            missing.append(f"Python {cls.PYTHON_VERSION}+")
            logger.error(f"Python version {platform.python_version()} is lower than required {cls.PYTHON_VERSION}")
        
        # 检查CUDA
        if cls.CUDA_HOME and not os.path.exists(cls.CUDA_HOME):
            missing.append("CUDA installation")
            logger.warning(f"CUDA directory not found at: {cls.CUDA_HOME}")
        
        return missing
    
    @classmethod
    def get_android_config(cls) -> dict:
        """获取Android相关配置"""
        return {
            'sdk_path': cls.ANDROID_HOME,
            'ndk_path': cls.ANDROID_NDK_HOME,
            'mtk_config': cls.MTK_CONFIG,
            'build_tools_version': '33.0.1',
            'compile_sdk_version': '33',
            'min_sdk_version': '24',
            'target_sdk_version': '33'
        }
    
    @classmethod
    def get_system_info(cls) -> dict:
        """获取系统信息"""
        return cls.SYSTEM_INFO
    
    @classmethod
    def setup_environment(cls):
        """设置环境变量"""
        os.environ['ANDROID_HOME'] = str(cls.ANDROID_HOME)
        os.environ['ANDROID_NDK_HOME'] = str(cls.ANDROID_NDK_HOME)
        
        if cls.CUDA_HOME:
            os.environ['CUDA_HOME'] = str(cls.CUDA_HOME)
            os.environ['CUDA_VISIBLE_DEVICES'] = cls.CUDA_VISIBLE_DEVICES
        
        # 设置Hugging Face缓存目录
        os.environ['HF_HOME'] = str(cls.HF_CACHE_DIR)
    
    @classmethod
    def print_environment_info(cls):
        """打印环境信息"""
        logger.info("Environment Information:")
        logger.info(f"Project Root: {cls.PROJECT_ROOT}")
        logger.info(f"Android SDK: {cls.ANDROID_HOME}")
        logger.info(f"Android NDK: {cls.ANDROID_NDK_HOME}")
        logger.info(f"Python Version: {platform.python_version()}")
        logger.info(f"CUDA Available: {cls.CUDA_HOME is not None}")
        if cls.CUDA_HOME:
            logger.info(f"CUDA Path: {cls.CUDA_HOME}")