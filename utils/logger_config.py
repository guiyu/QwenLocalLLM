# 文件路径: utils/logger_config.py

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from typing import Optional

class LoggerConfig:
    @staticmethod
    def setup_logger(name: str, log_file: Optional[str] = None, 
                    level: int = logging.INFO) -> logging.Logger:
        """配置日志器"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 清除已存在的处理器
        if logger.handlers:
            logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果指定）
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    @staticmethod
    def log_dict(logger: logging.Logger, data: dict, 
                 message: str = "Dictionary contents:"):
        """记录字典内容"""
        logger.info(f"{message}\n{json.dumps(data, indent=2)}")

    @staticmethod
    def log_model_info(logger: logging.Logger, model_info: dict):
        """记录模型信息"""
        logger.info("Model Information:")
        logger.info(f"Model path: {model_info.get('model_path', 'N/A')}")
        logger.info(f"Total size: {model_info.get('total_size', 0):.2f} MB")
        if 'files' in model_info:
            logger.info("Files:")
            for file in model_info['files']:
                logger.info(f"  - {file}")

    @staticmethod
    def log_performance(logger: logging.Logger, metrics: dict):
        """记录性能指标"""
        logger.info("Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.2f}")
            else:
                logger.info(f"{key}: {value}")

    @staticmethod
    def setup_error_logger(name: str, error_log_file: str) -> logging.Logger:
        """配置错误日志器"""
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Exception: %(exc_info)s\n'
            'Stack Trace: %(stack_info)s\n'
        )
        
        error_logger = logging.getLogger(f"{name}.error")
        error_logger.setLevel(logging.ERROR)
        
        # 创建错误日志文件处理器
        error_file_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_file_handler.setFormatter(error_formatter)
        error_logger.addHandler(error_file_handler)
        
        return error_logger

    @staticmethod
    def log_exception(logger: logging.Logger, e: Exception, 
                     context: str = ""):
        """记录异常详情"""
        import traceback
        error_details = {
            'type': type(e).__name__,
            'message': str(e),
            'context': context,
            'traceback': traceback.format_exc()
        }
        logger.error(
            f"Exception occurred{' in ' + context if context else ''}:",
            exc_info=True,
            extra={'error_details': error_details}
        )