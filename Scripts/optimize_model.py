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
        # 剪枝模型
        logger.info("Step 1: Pruning model")
        pruned_model_path = prune_model(
            model_path=config.QUANTIZED_MODEL_PATH,
            output_dir=config.PRUNED_MODEL_DIR,
            config=config
        )
        logger.info(f"Pruned model saved at: {pruned_model_path}")

        # 验证剪枝模型
        if not validate_model(pruned_model_path):
            logger.error(f"Pruned model is invalid: {pruned_model_path}")
            raise ValueError("Invalid pruned model!")

        # 量化模型
        logger.info("Step 2: Quantizing model")
        quantized_model_path = quantize_model(
            model_path=pruned_model_path,
            output_dir=config.QUANTIZED_MODEL_DIR,
            config=config
        )
        logger.info(f"Quantized model saved at: {quantized_model_path}")

        logger.info("Model optimization completed successfully!")
        return quantized_model_path
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise

def validate_model(model_path):
    import onnx
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"Model {model_path} is valid.")
        return True
    except onnx.checker.ValidationError as e:
        print(f"Model {model_path} is invalid: {e}")
        return False

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