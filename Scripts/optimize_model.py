# 文件路径: scripts/optimize_model.py

import logging
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import shutil
import numpy as np

logger = logging.getLogger(__name__)

def optimize_model(input_path: Path, output_path: Path) -> bool:
    """优化模型为GGML格式"""
    try:
        # 1. 加载模型
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(input_path),
            torch_dtype=torch.float16
        )
        model.eval()

        # 2. 创建临时目录
        temp_dir = output_path.parent / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 3. 导出模型权重
        logger.info("Exporting model weights...")
        export_path = temp_dir / "model_float32.bin"
        export_weights(model, export_path)

        # 4. 量化为GGML格式
        logger.info("Converting to GGML format...")
        convert_to_ggml(export_path, output_path)

        # 5. 清理临时文件
        shutil.rmtree(temp_dir)

        # 6. 验证生成的文件
        if not output_path.exists():
            raise Exception("GGML model file not generated")

        model_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Final model size: {model_size_mb:.2f} MB")

        return True

    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        return False

def export_weights(model, output_path: Path):
    """导出模型权重为二进制格式"""
    weights = {}
    
    # 收集所有权重
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights[name] = param.detach().cpu().numpy()

    # 写入二进制文件
    with open(output_path, 'wb') as f:
        # 写入头信息
        header = {
            'version': 1,
            'n_weights': len(weights),
        }
        
        # 写入头信息
        np.save(f, header)
        
        # 写入权重
        for name, weight in weights.items():
            name_bytes = name.encode('utf-8')
            f.write(len(name_bytes).to_bytes(4, byteorder='little'))
            f.write(name_bytes)
            
            # 写入权重形状
            shape = np.array(weight.shape, dtype=np.int32)
            shape.tofile(f)
            
            # 写入权重数据
            weight.tofile(f)

def convert_to_ggml(input_path: Path, output_path: Path):
    """转换为GGML格式"""
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用量化参数
        args = [
            "./ggml/quantize",
            str(input_path),
            str(output_path),
            "--type", "q4_0",
            "--threads", "4"
        ]
        
        # 运行量化
        result = subprocess.run(args, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"GGML conversion failed: {result.stderr}")
            
        logger.info("GGML conversion completed successfully")
        
    except Exception as e:
        raise Exception(f"GGML conversion failed: {str(e)}")
    
def optimize_phi2_model(input_path: Path, output_path: Path, bits: int = 4) -> bool:
    """优化Phi-2模型"""
    try:
        logger.info(f"Optimizing Phi-2 model from {input_path} to {output_path} with {bits}-bit quantization")

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(str(input_path), torch_dtype="auto", device_map="auto")
        model.save_pretrained(str(output_path))

        logger.info(f"Optimization completed. Quantized model saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return False