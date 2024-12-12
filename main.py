# 文件路径: main.py

import argparse
import sys
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
from scripts.optimize_model import optimize_phi2_model

logging.basicConfig(level=logging.DEBUG)


# 设置日志
logger = LoggerConfig.setup_logger(
    "QwenLocalLLM",
    log_file=str(project_root / "logs" / "qwen_local_llm.log")
)

class MobileLLMPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = Utils.get_available_device()
        self.deployment_manager = DeploymentManager(config)
        self.project_root = Path(__file__).parent.resolve()

        # 添加ASR和TTS路径
        self.model_dir = self.project_root / "models"
        self.android_dir = self.project_root / "android"
        self.asr_model_dir = self.model_dir / "whisper"
        self.tts_model_dir = self.model_dir / "fastspeech"

        # 创建必要的目录
        for dir_path in [
            self.model_dir / "original",
            self.model_dir / "quantized",
            self.model_dir / "android",
            self.asr_model_dir,
            self.tts_model_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def download_model(self):
        """下载模型"""
        logger.info("Downloading models...")
        from scripts.download_model import download_phi2_model, download_asr_model, download_tts_model
        
        try:
            # 1. 下载Phi-2模型
            model_path = self.model_dir / "original" / "phi-2"
            success = download_phi2_model(model_path)
            if not success:
                raise Exception("Phi-2 model download failed")
                
            # 2. 下载ASR模型
            success = download_asr_model(self.asr_model_dir)
            if not success:
                raise Exception("ASR model download failed")
                
            # 3. 下载TTS模型  
            success = download_tts_model(self.tts_model_dir)
            if not success:
                raise Exception("TTS model download failed")
                
            logger.info("All models downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False


    def optimize_model(self):
        """优化模型"""
        logger.info(f"Optimizing models...")
        from scripts.optimize_model import optimize_phi2_model
        
        try:
            # 1. 优化Phi-2模型为int4
            input_path = self.model_dir / "original" / "phi-2"
            output_path = self.model_dir / "quantized" / "model_quantized.onnx"
            success = optimize_phi2_model(input_path, output_path, bits=4) # 确保int4量化
            if not success:
                raise Exception("Phi-2 model optimization failed")

            # ASR和TTS模型已预优化，无需额外处理
                
            return True
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return False
        
    def deploy_android(self) -> bool:
        """部署模型到 Android"""
        logger.info("Starting Android deployment...")
        from scripts.deploy_android import deploy_to_android
        
        try:
            model_path = self.model_dir / "quantized" / "model_quantized.onnx"  # 假设这是量化后的模型路径
            android_dir = self.android_dir  # Android 项目路径
            
            success = deploy_to_android(model_path, android_dir)
            if not success:
                raise Exception("Android deployment failed")
            
            logger.info("Android deployment completed successfully")
            return True
        except Exception as e:
            logger.error(f"Android deployment failed: {e}")
            return False
        
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
    parser = argparse.ArgumentParser(description='Mobile LLM Deployment Tool')
    
    parser.add_argument(
        '--action',
        choices=['full', 'download', 'train', 'optimize', 'deploy'],
        default='full',
        help='Specify the action to perform'
    )

    parser.add_argument(
        '--basic',
        action='store_true',
        help='Use basic mode without optimizations'
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
def main():
    parser = argparse.ArgumentParser(description="Mobile LLM Deployment Tool")
    parser.add_argument(
        '--action',
        choices=['full', 'download', 'train', 'optimize', 'deploy'],
        default='full',  # 改为 'full' 而不是 'all'
        help='Action to perform'
    )
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Use basic mode with minimal optimizations'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Custom model path'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()
    
    # 导入配置
    from config.model_config import ModelConfig
    from config.env_config import EnvConfig
    
    # 创建pipeline时传入配置
    pipeline = MobileLLMPipeline(ModelConfig)
    
    try:
        if args.action in ['full', 'download']:
            if not pipeline.download_model():
                return 1
                
        if args.action in ['full', 'optimize']:
            if not pipeline.optimize_model():
                return 1
                
        if args.action in ['full', 'deploy']:
            if not pipeline.deploy_android():
                return 1
                
        logger.info("Pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
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