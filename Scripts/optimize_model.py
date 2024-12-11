# 文件路径: scripts/optimize_model.py

import logging
import torch
from pathlib import Path
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType
import shutil

logger = logging.getLogger(__name__)

def optimize_model(input_path: Path, output_path: Path) -> bool:
    """优化模型"""
    try:
        # 1. 加载模型
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(input_path),
            torch_dtype=torch.float16
        )
        model.eval()

        # 2. 导出为ONNX格式
        logger.info("Exporting to ONNX...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备输入
        dummy_input = torch.randint(0, 100, (1, 32), dtype=torch.long)
        
        # 导出配置
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
        
        # 导出模型
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=['input_ids'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=13
        )

        # 3. 验证ONNX模型
        logger.info("Validating ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # 4. 执行动态量化
        logger.info("Applying quantization...")
        quantized_path = str(output_path).replace('.onnx', '_quantized.onnx')
        quantize_dynamic(
            model_input=str(output_path),
            model_output=quantized_path,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=True
        )

        # 5. 使用量化后的模型替换原始模型
        shutil.move(quantized_path, output_path)

        # 6. 验证最终大小
        model_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Final model size: {model_size_mb:.2f} MB")

        return True

    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        return False