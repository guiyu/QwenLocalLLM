# 文件路径: utils/logger_config.py
# 新建文件

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

class LoggerConfig:
    @staticmethod
    def setup_logger(name, log_file=None, level=logging.INFO):
        """配置日志器"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建日志器
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
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
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger