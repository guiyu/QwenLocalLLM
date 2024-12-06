# 文件路径: utils/exceptions.py
# 新建文件

import sys  # 确保在这里也导入sys
import logging
import traceback
from functools import wraps

class QwenTTSError(Exception):
    """基础异常类"""
    pass

class ModelDownloadError(QwenTTSError):
    """模型下载错误"""
    pass

class DatasetError(QwenTTSError):
    """数据集相关错误"""
    pass

class ModelOptimizationError(QwenTTSError):
    """模型优化错误"""
    pass

class AndroidDeploymentError(QwenTTSError):
    """Android部署错误"""
    pass

logger = logging.getLogger(__name__)

def handle_exception(func):
    """异常处理装饰器"""
    @wraps(func)  # 保留原函数的元数据
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except QwenTTSError as e:
            # 获取调用栈信息
            stack_trace = traceback.format_exc()
            logger.error(f"Operation failed in {func.__name__}: {str(e)}\nStack trace:\n{stack_trace}")
            raise
        except Exception as e:
            # 获取详细的调用栈
            stack_trace = traceback.format_exc()
            logger.error(
                f"Unexpected error in {func.__name__}\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {str(e)}\n"
                f"Stack trace:\n{stack_trace}"
            )
            raise QwenTTSError(f"Unexpected error in {func.__name__}: {str(e)}")
    return wrapper