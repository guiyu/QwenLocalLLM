# 文件路径: scripts/optimize_model.py
# 新建文件

import sys
from pathlib import Path
import logging

# 添加项目根目录到环境变量
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from script.optimize.pruning_optimizer import prune_model
from script.optimize.quantization_optimizer import quantize_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_pipeline(config):
    try:
        # 1. 剪枝优化
        logger.info("Step 1: Model Pruning")
        pruned_model_path = prune_model(
            model_path=config.OUTPUT_DIR,
            output_dir=str(project_root / "models" / "pruned"),
            config=config
        )
        
        # 2. 量化优化
        logger.info("Step 2: Model Quantization")
        quantized_model_path = quantize_model(
            model_path=pruned_model_path,
            output_dir=str(project_root / "models" / "quantized"),
            config=config
        )
        
        logger.info("Model optimization completed!")
        logger.info(f"Final optimized model path: {quantized_model_path}")
        
        return quantized_model_path
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting model optimization process...")
        optimize_pipeline(ModelConfig)
        return True
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)