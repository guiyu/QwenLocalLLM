# 文件路径: scripts/deploy_android.py
# 新建文件

import sys
from pathlib import Path
import logging
import shutil
import argparse
# 添加项目根目录到环境变量
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from script.convert.android_converter import convert_for_android
from script.android.project_generator import generate_android_project

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_pipeline(config):
    try:
        # 1. 转换模型
        logger.info("Step 1: Converting model for Android")
        
        # 使用正确的量化模型目录路径
        quantized_model_dir = config.QUANTIZED_MODEL_DIR
        if not Path(quantized_model_dir).exists():
            raise FileNotFoundError(f"找不到量化模型目录: {quantized_model_dir}")
        
        conversion_paths = convert_for_android(
            model_path=quantized_model_dir,  # 传递量化模型所在目录
            output_dir=config.ANDROID_MODEL_DIR,
            config=config
        )
        
        # 2. 生成Android项目
        logger.info("Step 2: Generating Android project")
        project_paths = generate_android_project(config)
        
        return True
        
    except Exception as e:
        logger.error(f"部署过程中出错: {str(e)}")
        return False
    
def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Android Deployment Script')
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Use basic mode without optimizations'
    )
    return parser

def main():
    try:
        logger.info("Starting Android deployment process...")
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # 使用绝对路径并进行详细检查
        model_dir = project_root / "models" / "original"
        output_dir = project_root / "models" / "android"
        
        # 检查model_dir是否存在且包含必要的文件
        logger.info(f"Checking model directory: {model_dir}")
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        # 列出model_dir中的文件
        model_files = list(model_dir.glob('*'))
        if not model_files:
            raise FileNotFoundError(f"No files found in model directory: {model_dir}")
        
        logger.info(f"Found model files: {[f.name for f in model_files]}")
        
        # 执行完整的部署流程
        success = deploy_pipeline(ModelConfig)  # 添加这一行
        
        return success
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        return False
    
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)