# 文件路径: utils/logger_config.py

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

class LoggerConfig:
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
        """配置日志器"""
        # 创建logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # 清除已有的处理器
        logger.handlers.clear()

        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 如果指定了日志文件，创建文件处理器
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
    def log_info(logger: logging.Logger, message: str) -> None:
        """记录信息"""
        logger.info(message)

    @staticmethod
    def log_error(logger: logging.Logger, message: str, exc_info: bool = True) -> None:
        """记录错误"""
        logger.error(message, exc_info=exc_info)

    @staticmethod
    def log_debug(logger: logging.Logger, message: str) -> None:
        """记录调试信息"""
        logger.debug(message)