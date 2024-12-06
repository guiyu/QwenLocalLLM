# 文件路径: utils/exceptions.py
# 新建文件

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

def handle_exception(func):
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except QwenTTSError as e:
            logger.error(f"Operation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise QwenTTSError(f"Unexpected error: {str(e)}")
    return wrapper