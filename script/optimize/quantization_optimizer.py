# 文件路径: script/optimize/quantization_optimizer.py

import logging
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class QuantizationOptimizer:
    """模型量化优化器"""
    def __init__(self, config):
        self.config = config
        self.quantization_config = config.get_quantization_config()
        
    def quantize_model(self, model_path: Path) -> Optional[Path]:
        """量化模型"""
        try:
            logger.info(f"Starting model quantization for {model_path}")

                # 确保使用int4量化参数
            if self.quantization_config['bits'] != 4:
                logger.warning("Forcing 4-bit quantization as per requirement")
                self.quantization_config['bits'] = 4
                
            if self.quantization_config['method'] == 'ggml':
                return self._apply_ggml_quantization(model_path)
            else:
                return self._apply_dynamic_quantization(model_path)
                
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}")
            return None
            
    def _apply_ggml_quantization(self, model_path: Path) -> Optional[Path]:
        """应用GGML量化"""
        try:
            import ggml
            
            logger.info("Applying GGML quantization")
            output_path = self.config.QUANTIZED_MODEL_DIR / "model_ggml.bin"
            
            # 配置GGML量化参数
            params = {
                'bits': self.quantization_config['bits'],
                'groupsize': self.quantization_config['groupsize'],
                'use_sparse': self.quantization_config['use_sparse']
            }
            
            # 加载模型并量化
            model = ggml.load_model(str(model_path))
            quantized_model = ggml.quantize(model, **params)
            
            # 保存量化后的模型
            ggml.save_model(quantized_model, str(output_path))
            
            # 验证量化结果
            if not self._verify_ggml_model(output_path):
                raise Exception("GGML model verification failed")
                
            logger.info(f"GGML quantization completed, saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"GGML quantization failed: {str(e)}")
            return None
            
    def _apply_dynamic_quantization(self, model_path: Path) -> Optional[Path]:
        """应用动态量化"""
        try:
            logger.info("Applying dynamic quantization")
            output_path = self.config.QUANTIZED_MODEL_DIR / "model_quantized.onnx"
            
            # 配置量化参数
            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
                optimize_model=True,
                per_channel=False,
                reduce_range=True,
                extra_options={'ActivationSymmetric': True}
            )
            
            # 验证量化后的模型
            if not self._verify_onnx_model(output_path):
                raise Exception("ONNX model verification failed")
                
            logger.info(f"Dynamic quantization completed, saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {str(e)}")
            return None
            
    def _verify_ggml_model(self, model_path: Path) -> bool:
        """验证GGML模型"""
        try:
            import ggml
            
            # 加载模型
            model = ggml.load_model(str(model_path))
            
            # 准备测试输入
            test_input = np.random.randint(0, 100, (1, 32)).astype(np.int32)
            
            # 运行推理
            _ = model.run(test_input)
            
            # 检查模型大小
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"GGML model size: {model_size:.2f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"GGML model verification failed: {str(e)}")
            return False
            
    def _verify_onnx_model(self, model_path: Path) -> bool:
        """验证ONNX模型"""
        try:
            import onnxruntime as ort
            
            # 加载并检查模型
            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)
            
            # 创建推理会话
            session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            
            # 准备测试输入
            input_name = session.get_inputs()[0].name
            input_shape = (1, 32)
            input_data = np.random.randint(0, 100, input_shape).astype(np.int64)
            
            # 运行推理
            _ = session.run(None, {input_name: input_data})
            
            # 检查模型大小
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"ONNX model size: {model_size:.2f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {str(e)}")
            return False
            
    def benchmark_model(self, model_path: Path, num_runs: int = 10):
        """模型性能基准测试"""
        try:
            import onnxruntime as ort
            import time
            
            session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            
            input_name = session.get_inputs()[0].name
            input_data = np.random.randint(0, 100, (1, 32)).astype(np.int64)
            
            # 预热运行
            _ = session.run(None, {input_name: input_data})
            
            # 计时运行
            latencies = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = session.run(None, {input_name: input_data})
                latencies.append((time.time() - start_time) * 1000)  # 转换为毫秒
            
            # 计算统计信息
            avg_latency = sum(latencies) / len(latencies)
            p90_latency = np.percentile(latencies, 90)
            p99_latency = np.percentile(latencies, 99)
            
            logger.info("Benchmark Results:")
            logger.info(f"Average Latency: {avg_latency:.2f}ms")
            logger.info(f"P90 Latency: {p90_latency:.2f}ms")
            logger.info(f"P99 Latency: {p99_latency:.2f}ms")
            
            return {
                'average_latency': avg_latency,
                'p90_latency': p90_latency,
                'p99_latency': p99_latency
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            return None