# 文件路径: scripts/deploy_android.py
# 新建文件

import sys
from pathlib import Path
import logging
import shutil

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

def main():
    try:
        logger.info("Starting Android deployment process...")
        success = deploy_pipeline(ModelConfig)
        return success
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)