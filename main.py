# 文件路径: main.py

import argparse
import sys  # 添加这行
import logging
from pathlib import Path
from typing import Optional
import traceback
from functools import wraps

# 添加项目根目录到环境变量
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入配置和工具
from config.model_config import ModelConfig
from config.env_config import EnvConfig
from utils.logger_config import LoggerConfig
from utils.exceptions import QwenTTSError, handle_exception
from utils.helpers import Utils
from script.deploy.deployment_manager import DeploymentManager

# 设置日志
logger = LoggerConfig.setup_logger(
    "QwenLocalLLM",
    log_file=str(project_root / "logs" / "qwen_local_llm.log")
)

class QwenTTSPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = Utils.get_available_device()
        self.deployment_manager = DeploymentManager(config)
        
    @handle_exception
    def validate_environment(self) -> bool:
        """验证环境配置"""
        logger.info("Validating environment...")
        missing = EnvConfig.validate_environment()
        
        if missing:
            logger.error(f"Missing environment variables: {', '.join(missing)}")
            return False
            
        logger.info("Environment validation passed")
        return True
    
    @handle_exception
    def prepare_workspace(self) -> bool:
        """准备工作目录"""
        logger.info("Preparing workspace...")
        
        # 创建必要的目录
        directories = [
            "logs",
            "models/original",
            "models/pruned",
            "models/quantized",
            "models/android",
            "data/tts_dataset",
            "outputs"
        ]
        
        for dir_path in directories:
            (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Workspace preparation completed")
        return True
    
    @handle_exception
    def run_pipeline(self) -> bool:
        """运行完整处理流程"""
        if not self.validate_environment():
            return False
        
        if not self.prepare_workspace():
            return False
        
        return self.deployment_manager.run_full_pipeline()

def setup_argument_parser() -> argparse.ArgumentParser:
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='Qwen TTS Deployment Tool')
    
    parser.add_argument(
        '--action',
        choices=['full', 'download', 'train', 'optimize', 'deploy'],
        default='full',
        help='Specify the action to perform'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to the model directory'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser

@handle_exception
def main() -> int:
    """主函数"""
    try:
        # 解析命令行参数
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # 设置日志级别
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        # 创建pipeline实例
        pipeline = QwenTTSPipeline(ModelConfig)
        
        # 根据action执行相应操作
        if args.action == 'full':
            success = pipeline.run_pipeline()
        else:
            # 执行特定步骤
            action_map = {
                'download': pipeline.deployment_manager._run_script,
                'train': pipeline.deployment_manager._run_script,
                'optimize': pipeline.deployment_manager._run_script,
                'deploy': pipeline.deployment_manager._run_script
            }
            
            script_map = {
                'download': 'scripts/run_download.py',
                'train': 'scripts/train_model.py',
                'optimize': 'scripts/optimize_model.py',
                'deploy': 'scripts/deploy_android.py'
            }
            
            success = action_map[args.action](script_map[args.action])
        
        if success:
            logger.info("Operation completed successfully!")
            return 0
        else:
            logger.error("Operation failed!")
            return 1
            
    except QwenTTSError as e:
        logger.error(f"Operation failed with QwenTTSError: {str(e)}\n"
                    f"Context: action={args.action if 'args' in locals() else 'unknown'}\n"
                    f"Stack trace:\n{traceback.format_exc()}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error occurred:\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error message: {str(e)}\n"
                    f"Context: action={args.action if 'args' in locals() else 'unknown'}\n"
                    f"Stack trace:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    try:
        status = main()
        sys.exit(status)
    except Exception as e:
        logger.error(f"Critical error in main program:\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error message: {str(e)}\n"
                    f"Stack trace:\n{traceback.format_exc()}")
        sys.exit(1)