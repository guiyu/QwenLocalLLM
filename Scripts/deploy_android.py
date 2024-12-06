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
        conversion_paths = convert_for_android(
            model_path=config.QUANTIZED_MODEL_PATH,
            output_dir=config.ANDROID_MODEL_DIR,
            config=config
        )
        
        # 2. 生成Android项目
        logger.info("Step 2: Generating Android project")
        project_paths = generate_android_project(config)
        
        # 3. 复制模型和相关文件到Android项目
        logger.info("Step 3: Copying assets to Android project")
        assets_dir = Path(project_paths["assets_dir"])
        shutil.copytree(
            conversion_paths["assets_path"],
            assets_dir,
            dirs_exist_ok=True
        )
        
        # 4. 复制Java文件
        java_dir = Path(project_paths["java_dir"])
        shutil.copytree(
            conversion_paths["java_path"],
            java_dir,
            dirs_exist_ok=True
        )
        
        logger.info("Android deployment completed successfully!")
        logger.info(f"Android project directory: {config.ANDROID_OUTPUT_DIR}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during deployment: {str(e)}")
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
        
        # 进行转换
        success = convert_for_android(
            model_path=str(model_dir),
            output_dir=str(output_dir),
            config=ModelConfig,
            basic_mode=args.basic
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        return False
    
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)