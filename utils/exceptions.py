# 文件路径: utils/exceptions.py

import logging
from functools import wraps

logger = logging.getLogger(__name__)

class QwenTTSError(Exception):
    """基础异常类"""
    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ModelError(QwenTTSError):
    """模型相关错误"""
    pass

class DownloadError(QwenTTSError):
    """下载错误"""
    pass

class OptimizationError(QwenTTSError):
    """优化错误"""
    pass

class DeploymentError(QwenTTSError):
    """部署错误"""
    pass

def handle_exception(func):
    """异常处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except QwenTTSError as e:
            logger.error(f"Business error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"System error in {func.__name__}: {str(e)}")
            raise QwenTTSError(f"Unexpected error in {func.__name__}: {str(e)}")
    return wrapper