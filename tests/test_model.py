# 文件路径: tests/test_model.py

import unittest
import sys
from pathlib import Path
import torch
import logging
import shutil
import tempfile
import numpy as np
import onnxruntime as ort

# 添加项目根目录到环境变量
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from config.env_config import EnvConfig
from script.download.model_downloader import ModelDownloader
from script.optimize.model_optimizer import ModelOptimizer
from script.optimize.quantization_optimizer import QuantizationOptimizer
from script.optimize.memory_optimizer import MemoryOptimizer

class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        
        # 创建临时目录
        cls.temp_dir = tempfile.mkdtemp()
        cls.config = ModelConfig
        
        # 修改配置使用临时目录
        cls.config.MODEL_DIR = Path(cls.temp_dir) / "models"
        cls.config.MODEL_DIR.mkdir(parents=True)
        
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        shutil.rmtree(cls.temp_dir)
        
    def setUp(self):
        """每个测试用例的设置"""
        self.model_downloader = ModelDownloader(self.config)
        self.model_optimizer = ModelOptimizer(self.config)
        self.quantization_optimizer = QuantizationOptimizer(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        
    def test_model_download(self):
        """测试模型下载"""
        result = self.model_downloader.download_all_models()
        self.assertTrue(result)
        
        # 验证文件存在
        model_path = self.config.ORIGINAL_MODEL_DIR / "phi-2"
        self.assertTrue(model_path.exists())
        
        # 验证必要文件
        required_files = ["config.json", "model.safetensors", "tokenizer.json"]
        for file in required_files:
            self.assertTrue((model_path / file).exists())
            
    def test_model_optimization(self):
        """测试模型优化"""
        # 首先下载模型
        self.model_downloader.download_all_models()
        
        # 测试优化过程
        result = self.model_optimizer.optimize_all_models()
        self.assertTrue(result)
        
        # 验证优化后的文件
        optimized_model = self.config.QUANTIZED_MODEL_DIR / "model.onnx"
        self.assertTrue(optimized_model.exists())
        
        # 验证模型大小满足要求
        model_size = optimized_model.stat().st_size / (1024 * 1024)  # MB
        self.assertLess(model_size, self.config.PERFORMANCE_TARGETS['storage_size'])
        
    def test_model_quantization(self):
        """测试模型量化"""
        # 准备测试模型
        self.model_downloader.download_all_models()
        self.model_optimizer.optimize_all_models()
        
        # 测试量化
        model_path = self.config.get_model_paths()['original']
        quantized_path = self.quantization_optimizer.quantize_model(model_path)
        self.assertIsNotNone(quantized_path)
        self.assertTrue(Path(quantized_path).exists())
        
        # 测试量化后的性能
        benchmark_results = self.quantization_optimizer.benchmark_model(
            Path(quantized_path))
        self.assertIsNotNone(benchmark_results)
        self.assertLess(
            benchmark_results['average_latency'],
            self.config.PERFORMANCE_TARGETS['inference_time'] * 1000  # 转换为毫秒
        )
        
    def test_memory_optimization(self):
        """测试内存优化"""
        result = self.memory_optimizer.optimize_memory_usage()
        self.assertTrue(result)
        
        # 验证内存配置
        memory_config = self.memory_optimizer.get_memory_config()
        self.assertIsNotNone(memory_config)
        
        # 测试内存监控
        memory_stats = self.memory_optimizer.monitor_memory_usage()
        self.assertIsNotNone(memory_stats)
        self.assertIn('rss', memory_stats)
        self.assertLess(
            memory_stats['rss'],
            self.config.PERFORMANCE_TARGETS['memory_usage']
        )
        
    def test_model_inference(self):
        """测试模型推理"""
        # 准备模型
        self.model_downloader.download_all_models()
        self.model_optimizer.optimize_all_models()
        quantized_path = self.config.get_model_paths()['quantized_model']
        
        # 创建推理会话
        session = ort.InferenceSession(
            str(quantized_path),
            providers=['CPUExecutionProvider']
        )
        
        # 准备测试输入
        input_name = session.get_inputs()[0].name
        input_shape = (1, 32)
        input_data = np.random.randint(0, 100, input_shape).astype(np.int64)
        
        # 运行推理
        import time
        start_time = time.time()
        outputs = session.run(None, {input_name: input_data})
        inference_time = time.time() - start_time
        
        # 验证输出
        self.assertIsNotNone(outputs)
        self.assertTrue(len(outputs) > 0)
        
        # 验证推理时间
        self.assertLess(
            inference_time,
            self.config.PERFORMANCE_TARGETS['inference_time']
        )
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效模型路径
        with self.assertRaises(Exception):
            self.quantization_optimizer.quantize_model(Path("invalid/path"))
            
        # 测试内存超限处理
        self.memory_optimizer.memory_config['total_limit'] = 1  # 设置极小的内存限制
        memory_stats = self.memory_optimizer.monitor_memory_usage()
        self.assertIn('rss', memory_stats)  # 应该仍然能获取内存统计
        
if __name__ == '__main__':
    unittest.main()