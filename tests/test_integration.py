# 文件路径: tests/test_integration.py

import unittest
import sys
from pathlib import Path
import shutil
import tempfile
import logging
import time
import json

# 添加项目根目录到环境变量
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from script.deploy.deployment_manager import DeploymentManager

class TestIntegration(unittest.TestCase):
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
        
        # 创建部署管理器
        cls.deployment_manager = DeploymentManager(cls.config)
        
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        shutil.rmtree(cls.temp_dir)
        
    def setUp(self):
        """每个测试前的设置"""
        # 记录开始时间
        self.start_time = time.time()
        
    def tearDown(self):
        """每个测试后的清理"""
        # 记录测试用时
        duration = time.time() - self.start_time
        print(f"\nTest duration: {duration:.2f} seconds")
        
    def test_full_pipeline(self):
        """测试完整部署流程"""
        # 运行完整流程
        result = self.deployment_manager.run_full_pipeline()
        self.assertTrue(result)
        
        # 验证各个阶段的输出
        self._verify_model_files()
        self._verify_android_files()
        self._verify_performance()
        
    def test_basic_mode(self):
        """测试基础模式部署"""
        result = self.deployment_manager.run_full_pipeline(basic_mode=True)
        self.assertTrue(result)
        
        # 验证基础模式的输出
        self._verify_model_files(basic_mode=True)
        self._verify_android_files()
        
    def test_incremental_deployment(self):
        """测试增量部署"""
        # 1. 首先下载
        result = self.deployment_manager.run_download()
        self.assertTrue(result)
        self._verify_downloaded_files()
        
        # 2. 然后优化
        result = self.deployment_manager.run_optimize()
        self.assertTrue(result)
        self._verify_optimized_files()
        
        # 3. 最后部署
        result = self.deployment_manager.run_deploy()
        self.assertTrue(result)
        self._verify_android_files()
        
    def _verify_model_files(self, basic_mode=False):
        """验证模型文件"""
        # 检查必要的模型文件
        paths = self.config.get_model_paths()
        
        # 原始模型
        self.assertTrue((paths['original'] / "phi-2" / "config.json").exists())
        
        # 量化模型
        if not basic_mode:
            self.assertTrue(paths['quantized_model'].exists())
        
        # Android模型
        self.assertTrue((paths['android'] / "model_quantized.onnx").exists())
        
    def _verify_android_files(self):
        """验证Android项目文件"""
        android_dir = project_root / "android"
        
        # 检查必要的Android项目文件
        required_files = [
            "app/build.gradle",
            "app/src/main/AndroidManifest.xml",
            "app/src/main/assets/models/model_quantized.onnx"
        ]
        
        for file in required_files:
            self.assertTrue((android_dir / file).exists())
            
    def _verify_performance(self):
        """验证性能指标"""
        # 获取性能指标
        model_path = self.config.get_model_paths()['quantized_model']
        
        # 检查模型大小
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        self.assertLess(model_size, self.config.PERFORMANCE_TARGETS['storage_size'])
        
        # 检查推理性能
        inference_results = self._run_inference_test()
        self.assertLess(
            inference_results['average_latency'],
            self.config.PERFORMANCE_TARGETS['inference_time'] * 1000
        )
        
    def _verify_downloaded_files(self):
        """验证下载的文件"""
        paths = self.config.get_model_paths()
        model_path = paths['original'] / "phi-2"
        
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        for file in required_files:
            self.assertTrue((model_path / file).exists())
            
    def _verify_optimized_files(self):
        """验证优化后的文件"""
        paths = self.config.get_model_paths()
        
        # 检查量化模型
        self.assertTrue(paths['quantized_model'].exists())
        
        # 检查模型大小
        model_size = paths['quantized_model'].stat().st_size / (1024 * 1024)
        self.assertLess(model_size, self.config.PERFORMANCE_TARGETS['storage_size'])
        
    def _run_inference_test(self):
        """运行推理测试"""
        import onnxruntime as ort
        import numpy as np
        
        model_path = self.config.get_model_paths()['quantized_model']
        
        # 创建推理会话
        session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        
        # 准备输入
        input_name = session.get_inputs()[0].name
        input_data = np.random.randint(0, 100, (1, 32)).astype(np.int64)
        
        # 运行多次推理
        latencies = []
        num_runs = 10
        
        # 预热
        _ = session.run(None, {input_name: input_data})
        
        # 测试运行
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {input_name: input_data})
            latencies.append((time.time() - start_time) * 1000)  # 转换为毫秒
            
        return {
            'average_latency': sum(latencies) / len(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'p90_latency': sorted(latencies)[int(0.9 * len(latencies))]
        }

if __name__ == '__main__':
    unittest.main()