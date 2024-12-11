# 文件路径: utils/exceptions.py

import sys
import logging
import traceback
from functools import wraps
from typing import Type, Callable, Any, Optional, Dict

logger = logging.getLogger(__name__)

class MobileLLMError(Exception):
    """基础异常类"""
    def __init__(self, message: str, error_code: Optional[int] = None, 
                 details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class ModelError(MobileLLMError):
    """模型相关错误"""
    def __init__(self, message: str, model_name: str = "", 
                 error_code: Optional[int] = None):
        details = {'model_name': model_name}
        super().__init__(message, error_code, details)

class DownloadError(MobileLLMError):
    """下载错误"""
    def __init__(self, message: str, url: str = "", 
                 error_code: Optional[int] = None):
        details = {'url': url}
        super().__init__(message, error_code, details)

class OptimizationError(MobileLLMError):
    """优化错误"""
    def __init__(self, message: str, optimization_type: str = "", 
                 error_code: Optional[int] = None):
        details = {'optimization_type': optimization_type}
        super().__init__(message, error_code, details)

class DeploymentError(MobileLLMError):
    """部署错误"""
    def __init__(self, message: str, deployment_stage: str = "", 
                 error_code: Optional[int] = None):
        details = {'deployment_stage': deployment_stage}
        super().__init__(message, error_code, details)

class ResourceError(MobileLLMError):
    """资源错误"""
    def __init__(self, message: str, resource_type: str = "", 
                 error_code: Optional[int] = None):
        details = {'resource_type': resource_type}
        super().__init__(message, error_code, details)

class InvalidConfigError(MobileLLMError):
    """配置错误"""
    def __init__(self, message: str, config_key: str = "", 
                 error_code: Optional[int] = None):
        details = {'config_key': config_key}
        super().__init__(message, error_code, details)

def handle_exception(func: Callable) -> Callable:
    """异常处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except MobileLLMError as e:
            # 业务异常处理
            logger.error(
                f"Business error in {func.__name__}: {str(e)}\n"
                f"Error code: {e.error_code}\n"
                f"Details: {e.details}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )
            raise
        except Exception as e:
            # 系统异常处理
            logger.error(
                f"System error in {func.__name__}\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {str(e)}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )
            raise MobileLLMError(
                f"Unexpected error in {func.__name__}: {str(e)}"
            )
    return wrapper

class ErrorHandler:
    """错误处理器"""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def handle(self, e: Exception, context: str = "") -> None:
        """处理异常"""
        error_details = {
            'type': type(e).__name__,
            'message': str(e),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        if isinstance(e, MobileLLMError):
            error_details.update({
                'error_code': e.error_code,
                'details': e.details
            })
        
        self.logger.error(
            f"Error occurred{' in ' + context if context else ''}:",
            exc_info=True,
            extra={'error_details': error_details}
        )
        
    def handle_and_exit(self, e: Exception, context: str = "", 
                       exit_code: int = 1) -> None:
        """处理异常并退出程序"""
        self.handle(e, context)
        sys.exit(exit_code)
        
    def create_error_report(self, e: Exception, context: str = "") -> Dict:
        """创建错误报告"""
        report = {
            'timestamp': Utils.get_current_time(),
            'error_type': type(e).__name__,
            'message': str(e),
            'context': context,
            'traceback': traceback.format_exc(),
            'system_info': Utils.get_system_info()
        }
        
        if isinstance(e, MobileLLMError):
            report.update({
                'error_code': e.error_code,
                'details': e.details
            })
            
        return report
    
    def save_error_report(self, report: Dict, 
                         filepath: Optional[str] = None) -> None:
        """保存错误报告"""
        if filepath is None:
            filepath = f"error_report_{Utils.get_current_time()}.json"
            
        try:
            Utils.save_json(report, filepath)
            self.logger.info(f"Error report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
            
class ErrorRegistry:
    """错误注册表"""
    _error_codes = {}
    _handlers = {}
    
    @classmethod
    def register_error(cls, error_code: int, error_class: Type[MobileLLMError], 
                      handler: Optional[Callable] = None) -> None:
        """注册错误代码和处理器"""
        cls._error_codes[error_code] = error_class
        if handler:
            cls._handlers[error_code] = handler
            
    @classmethod
    def get_error_class(cls, error_code: int) -> Optional[Type[MobileLLMError]]:
        """获取错误类"""
        return cls._error_codes.get(error_code)
        
    @classmethod
    def get_handler(cls, error_code: int) -> Optional[Callable]:
        """获取错误处理器"""
        return cls._handlers.get(error_code)
        
    @classmethod
    def handle_error(cls, error_code: int, *args, **kwargs) -> None:
        """处理特定错误"""
        handler = cls.get_handler(error_code)
        if handler:
            handler(*args, **kwargs)
        else:
            error_class = cls.get_error_class(error_code)
            if error_class:
                raise error_class(*args, **kwargs)
            else:
                raise MobileLLMError(f"Unknown error code: {error_code}")

# 注册常见错误代码
ErrorRegistry.register_error(1001, ModelError)
ErrorRegistry.register_error(2001, DownloadError)
ErrorRegistry.register_error(3001, OptimizationError)
ErrorRegistry.register_error(4001, DeploymentError)
ErrorRegistry.register_error(5001, ResourceError)
ErrorRegistry.register_error(6001, InvalidConfigError)