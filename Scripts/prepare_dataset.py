# 文件路径: scripts/prepare_dataset.py
# 新建文件

import sys
from pathlib import Path
import logging

# 添加项目根目录到环境变量
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from script.data.dataset_utils import TTSDatasetBuilder
from config.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("开始生成测试数据集...")
        
        # 创建数据集生成器
        dataset_builder = TTSDatasetBuilder(
            output_dir=str(project_root / "data" / "tts_dataset")
        )
        
        # 生成测试数据
        dataset_path = dataset_builder.generate_test_data(num_samples=1000)
        
        logger.info(f"数据集生成完成，保存在: {dataset_path}")
        return True
        
    except Exception as e:
        logger.error(f"数据集生成过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)