# 文件路径: scripts/run_download.py
# 新建文件

import argparse
import sys  # 添加这行
import os
import logging
from pathlib import Path
from typing import Optional

# 添加项目根目录到环境变量
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from script.download.model_downloader import download_qwen_model  # 注意这里是 script 不是 scripts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("开始模型下载流程...")
        
        # 创建必要的目录
        os.makedirs(ModelConfig.CACHE_DIR, exist_ok=True)
        os.makedirs(ModelConfig.OUTPUT_DIR, exist_ok=True)
        
        # 下载模型
        model, tokenizer = download_qwen_model(
            model_version=ModelConfig.MODEL_VERSION,
            output_dir=ModelConfig.OUTPUT_DIR
        )
        
        logger.info("模型下载完成！")
        return True
        
    except Exception as e:
        logger.error(f"下载过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)