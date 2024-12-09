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
from onnx import TensorProto
import numpy as np
import traceback

logger = logging.getLogger(__name__)

class ModelQuantizer:
    def __init__(self, model_path, output_dir="./models/quantized"):
            self.model_path = Path(model_path)
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.model = None  # 初始化为 None
            logger.info(f"Initialized quantizer with model path: {self.model_path}")
            logger.info(f"Output directory: {self.output_dir}")
        
    def load_model(self):
            """加载模型"""
            logger.info(f"Loading model from {self.model_path}")
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.eval()  # 设置为评估模式
            return self
        
    def export_to_onnx(self):
        """导出为ONNX格式"""
        logger.info("Exporting model to ONNX format")
        try:
            # 创建一个模型包装器来处理注意力掩码
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids):
                    # 创建注意力掩码
                    attention_mask = torch.ones_like(input_ids)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    return outputs.logits

            # 包装模型
            wrapped_model = ModelWrapper(self.model)
            wrapped_model.eval()

            # 准备示例输入
            dummy_input = torch.randint(0, 100, (1, 32), dtype=torch.long)
            
            onnx_path = self.output_dir / "model.onnx"
            
            # 导出到ONNX
            torch.onnx.export(
                wrapped_model,
                (dummy_input,),  # 只传入 input_ids
                str(onnx_path),
                input_names=['input_ids'],  # 简化输入名称
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'}
                },
                opset_version=14,
                do_constant_folding=True
            )
            
            logger.info(f"Exported model to {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def dynamic_quantization(self, onnx_path):
        """动态量化"""
        logger.info("Applying dynamic quantization")
        
        quantized_path = self.output_dir / "model_quantized.onnx"
        
        # 修改这里：移除 optimize_model 参数
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=True,  # 可以提高某些硬件上的性能
            extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT}
        )
        
        logger.info(f"Quantized model saved to {quantized_path}")
        return str(quantized_path)
    
    def verify_quantized_model(self, quantized_path):
        """验证量化后的模型"""
        logger.info("Verifying quantized model")
        try:
            import onnxruntime as ort
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 加载并检查模型
            session = ort.InferenceSession(
                quantized_path,
                session_options,
                providers=['CPUExecutionProvider']
            )
            
            # 获取输入详情
            input_details = session.get_inputs()
            logger.info("Model input details:")
            for input in input_details:
                logger.info(f"  Name: {input.name}")
                logger.info(f"  Shape: {input.shape}")
                logger.info(f"  Type: {input.type}")
            
            # 准备输入
            input_feed = {
                'input_ids': np.random.randint(0, 100, (1, 32), dtype=np.int64)
            }
            
            # 执行推理
            logger.info("Running inference...")
            outputs = session.run(None, input_feed)
            
            logger.info("Quantized model verification successful!")
            logger.info(f"Output shapes: {[out.shape for out in outputs]}")
            
            # 不返回 bool 值，而是在成功时不抛出异常
            
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            raise

# def quantize_model(model_path, output_dir, config):
#     """主量化函数"""
#     try:
#         logger.info("Starting model quantization process...")
        
#         quantizer = ModelQuantizer(model_path, output_dir)
        
#         # 执行量化流程
#         # 1. 加载模型
#         quantizer.load_model()
        
#         # 2. 导出为ONNX
#         onnx_path = quantizer.export_to_onnx()
#         if not onnx_path:
#             raise ValueError("ONNX export failed")
            
#         # 3. 执行动态量化
#         quantized_path = quantizer.dynamic_quantization(onnx_path)
#         if not quantized_path:
#             raise ValueError("Quantization failed")
            
#         # 4. 验证量化后的模型
#         try:
#             quantizer.verify_quantized_model(quantized_path)
#             logger.info("Model quantization completed successfully!")
#         except Exception as e:
#             logger.error(f"Model verification failed but model was quantized: {e}")
        
#         return quantized_path
            
#     except Exception as e:
#         logger.error(f"Error during quantization: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise
def quantize_model(model_path, output_dir, config):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from pathlib import Path

    # 定义量化后的模型路径
    quantized_model_path = Path(output_dir) / "quantized_android_model.onnx"

    # 动态量化模型
    quantize_dynamic(
        model_input=model_path,
        model_output=str(quantized_model_path),
        weight_type=QuantType.QUInt8  # 使用8位量化
    )

    print(f"Quantized model saved to: {quantized_model_path}")
    return quantized_model_path