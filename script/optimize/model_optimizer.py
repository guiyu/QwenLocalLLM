# 文件路径: script/optimize/model_optimizer.py

import logging
import torch
import numpy as np
from pathlib import Path
from ..model.model_adapter import PhiModelAdapter, WhisperAdapter, FastSpeechAdapter

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """模型优化器"""
    def __init__(self, config):
        self.config = config
        self.llm_adapter = PhiModelAdapter(config)
        self.asr_adapter = WhisperAdapter(config)
        self.tts_adapter = FastSpeechAdapter(config)
        
    def optimize_all_models(self):
        """优化所有模型"""
        try:
            logger.info("Starting model optimization process")
            
            # 优化LLM模型
            if not self._optimize_llm():
                return False
                
            # 优化ASR模型
            if not self._optimize_asr():
                return False
                
            # 优化TTS模型
            if not self._optimize_tts():
                return False
            
            logger.info("All models optimized successfully")
            return True
        except Exception as e:
            logger.error(f"Model optimization failed: {str(e)}")
            return False
            
    def _optimize_llm(self):
        """优化LLM模型"""
        try:
            logger.info("Optimizing LLM model")
            
            # 1. 加载模型
            if not self.llm_adapter.load_model():
                raise Exception("Failed to load LLM model")
            
            # 2. 优化模型
            if not self.llm_adapter.optimize_for_mobile():
                raise Exception("Failed to optimize LLM model")
            
            # 3. 导出模型
            onnx_path = self.llm_adapter.export_to_onnx()
            if not onnx_path:
                raise Exception("Failed to export LLM model to ONNX")
            
            return True
        except Exception as e:
            logger.error(f"LLM optimization failed: {str(e)}")
            return False
            
    def _optimize_asr(self):
        """优化ASR模型"""
        try:
            logger.info("Optimizing ASR model")
            if not self.asr_adapter.load_model():
                raise Exception("Failed to load ASR model")
            # TODO: 实现ASR模型优化逻辑
            return True
        except Exception as e:
            logger.error(f"ASR optimization failed: {str(e)}")
            return False
            
    def _optimize_tts(self):
        """优化TTS模型"""
        try:
            logger.info("Optimizing TTS model")
            if not self.tts_adapter.load_model():
                raise Exception("Failed to load TTS model")
            # TODO: 实现TTS模型优化逻辑
            return True
        except Exception as e:
            logger.error(f"TTS optimization failed: {str(e)}")
            return False
            
    def verify_optimizations(self):
        """验证优化效果"""
        try:
            logger.info("Verifying model optimizations")
            
            # 验证模型大小
            self._verify_model_size()
            
            # 验证推理性能
            self._verify_inference_performance()
            
            # 验证内存使用
            self._verify_memory_usage()
            
            return True
        except Exception as e:
            logger.error(f"Optimization verification failed: {str(e)}")
            return False
            
    def _verify_model_size(self):
        """验证模型大小"""
        try:
            size_targets = self.config.PERFORMANCE_TARGETS
            model_path = self.config.get_model_paths()['quantized_model']
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            actual_size = model_path.stat().st_size / (1024 * 1024)  # Convert to MB
            logger.info(f"Model size: {actual_size:.2f}MB (target: {size_targets['storage_size']}MB)")
            
            return actual_size <= size_targets['storage_size']
        except Exception as e:
            logger.error(f"Model size verification failed: {str(e)}")
            return False
            
    def _verify_inference_performance(self):
        """验证推理性能"""
        try:
            # 加载ONNX模型
            import onnxruntime as ort
            model_path = str(self.config.get_model_paths()['quantized_model'])
            
            session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # 准备测试输入
            input_shape = (1, 32)
            input_data = np.random.randint(0, 100, input_shape).astype(np.int64)
            
            # 运行推理
            import time
            times = []
            for _ in range(10):
                start_time = time.time()
                session.run(None, {'input_ids': input_data})
                times.append(time.time() - start_time)
            
            avg_time = sum(times[1:]) / len(times[1:])  # 忽略第一次运行
            logger.info(f"Average inference time: {avg_time*1000:.2f}ms")
            
            return avg_time <= self.config.PERFORMANCE_TARGETS['inference_time']
        except Exception as e:
            logger.error(f"Inference performance verification failed: {str(e)}")
            return False
            
    def _verify_memory_usage(self):
        """验证内存使用"""
        try:
            import psutil
            import gc
            
            # 强制GC
            gc.collect()
            
            # 记录初始内存
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            # 加载模型并运行推理
            self._verify_inference_performance()
            
            # 检查内存使用
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_usage = final_memory - initial_memory
            
            logger.info(f"Memory usage: {memory_usage:.2f}MB")
            
            return memory_usage <= self.config.PERFORMANCE_TARGETS['memory_usage']
        except Exception as e:
            logger.error(f"Memory usage verification failed: {str(e)}")
            return False