# 文件路径: script/verify/verify_model.py

import logging
import sys
from pathlib import Path
from onnx_verify import ONNXModelVerifier, verify_onnx_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        
        # 设置模型路径
        model_dir = project_root / "models" / "android"
        model_path = model_dir / "model.onnx"
        
        # 检查model.onnx是否存在
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return False
            
        logger.info(f"开始验证模型: {model_path}")
        
        # 创建验证器实例
        verifier = ONNXModelVerifier(str(model_path))
        
        # 修复模型
        logger.info("步骤1: 修复模型")
        fixed_model_path = verifier.fix_slice_inputs(str(model_path))
        if not fixed_model_path:
            logger.error("模型修复失败")
            return False
            
        logger.info(f"修复后的模型保存在: {fixed_model_path}")
        
        # 创建新的验证器实例使用修复后的模型
        verifier = ONNXModelVerifier(fixed_model_path)
        
        # 加载模型会话
        logger.info("步骤2: 加载模型会话")
        if not verifier.load_session():
            logger.error("加载模型会话失败")
            return False
            
        # 验证基本推理
        logger.info("步骤3: 验证基本推理")
        if not verifier.verify_basic_inference():
            logger.error("基本推理验证失败")
            return False
            
        logger.info("模型验证成功完成!")
        return True
        
    except Exception as e:
        logger.error(f"验证过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)