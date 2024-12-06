# 文件路径: script/optimize/quantization_optimizer.py
# 更新文件，完善量化策略

import torch
from transformers import AutoModelForCausalLM
import onnx
import onnxruntime as ort
from pathlib import Path
import logging
import json
from onnxruntime.quantization import quantize_dynamic, QuantType

logger = logging.getLogger(__name__)

class ModelQuantizer:
    def __init__(self, model_path, output_dir="./models/quantized"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """加载模型"""
        logger.info(f"Loading model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        return self
    
    def export_to_onnx(self):
        """导出为ONNX格式"""
        logger.info("Exporting model to ONNX format")
        
        # 准备示例输入
        dummy_input = torch.randint(100, (1, 64))
        onnx_path = self.output_dir / "model.onnx"
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            },
            opset_version=13,
            do_constant_folding=True
        )
        
        logger.info(f"Exported model to {onnx_path}")
        return str(onnx_path)
    
    def dynamic_quantization(self, onnx_path):
        """动态量化"""
        logger.info("Applying dynamic quantization")
        
        quantized_path = self.output_dir / "model_quantized.onnx"
        quantize_dynamic(
            onnx_path,
            str(quantized_path),
            weight_type=QuantType.QInt8,
            optimize_model=True,
            per_channel=False,
            reduce_range=True
        )
        
        return str(quantized_path)
    
    def verify_quantized_model(self, quantized_path):
        """验证量化后的模型"""
        logger.info("Verifying quantized model")
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            quantized_path,
            session_options,
            providers=['CPUExecutionProvider']
        )
        
        # 测试推理性能
        dummy_input = torch.randint(100, (1, 64)).numpy()
        input_data = {'input': dummy_input}
        
        import time
        times = []
        
        # 预热
        session.run(None, input_data)
        
        # 测试延迟
        for _ in range(10):
            start = time.time()
            session.run(None, input_data)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        
        # 保存性能信息
        performance_info = {
            "average_inference_time_ms": avg_time * 1000,
            "model_size_mb": Path(quantized_path).stat().st_size / (1024 * 1024)
        }
        
        with open(self.output_dir / "quantization_info.json", "w") as f:
            json.dump(performance_info, f, indent=2)
        
        return performance_info

def quantize_model(model_path, output_dir, config):
    try:
        logger.info("Starting model quantization process...")
        
        quantizer = ModelQuantizer(model_path, output_dir)
        
        # 执行量化流程
        onnx_path = quantizer.load_model().export_to_onnx()
        quantized_path = quantizer.dynamic_quantization(onnx_path)
        performance_info = quantizer.verify_quantized_model(quantized_path)
        
        logger.info("Model quantization completed successfully!")
        logger.info(f"Average inference time: {performance_info['average_inference_time_ms']:.2f}ms")
        logger.info(f"Quantized model size: {performance_info['model_size_mb']:.2f}MB")
        
        return quantized_path
        
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        raise