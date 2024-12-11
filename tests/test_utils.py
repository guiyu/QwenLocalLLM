# 文件路径: tests/test_utils.py

import unittest
import sys
from pathlib import Path
import tempfile
import shutil
import json
import time

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from utils.cache_manager import CacheManager
from utils.resource_manager import ResourceManager
from utils.config_validator import ConfigValidator
from utils.performance_monitor import PerformanceMonitor
from utils.exceptions import MobileLLMError, InvalidConfigError

class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.config = ModelConfig
        cls.config.MODEL_DIR = cls.temp_dir / "models"
        cls.config.MODEL_DIR.mkdir(parents=True)
        
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        shutil.rmtree(cls.temp_dir)
        
    def setUp(self):
        """每个测试用例的设置"""
        self.cache_manager = CacheManager(self.config)
        self.resource_manager = ResourceManager(self.config)
        self.config_validator = ConfigValidator(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        
    def test_cache_manager(self):
        """测试缓存管理器"""
        # 测试添加缓存
        test_data = {"key": "value"}
        success = self.cache_manager.add_to_cache("test_key", test_data)
        self.assertTrue(success)
        
        # 测试获取缓存
        cached_data = self.cache_manager.get_from_cache("test_key")
        self.assertEqual(cached_data, test_data)
        
        # 测试清理缓存
        success = self.cache_manager.remove_from_cache("test_key")
        self.assertTrue(success)
        
        # 验证缓存已清理
        cached_data = self.cache_manager.get_from_cache("test_key")
        self.assertIsNone(cached_data)
        
    def test_resource_manager(self):
        """测试资源管理器"""
        # 开始资源监控
        self.resource_manager.start_monitoring(interval=0.1)
        time.sleep(0.2)  # 等待监控数据收集
        
        # 测试资源使用情况
        resource_usage = self.resource_manager.get_resource_usage()
        self.assertIn('cpu', resource_usage)
        self.assertIn('memory', resource_usage)
        self.assertIn('disk', resource_usage)
        
        # 测试资源报告生成
        report = self.resource_manager.generate_resource_report()
        self.assertIn('resources', report)
        self.assertIn('warnings', report)
        self.assertIn('recommendations', report)
        
        # 停止监控
        self.resource_manager.stop_monitoring()
        
    def test_config_validator(self):
        """测试配置验证器"""
        # 测试完整验证
        validation_result = self.config_validator.validate_all()
        self.assertTrue(validation_result)
        
        # 测试验证报告生成
        report = self.config_validator.generate_validation_report()
        self.assertIn('validation_status', report)
        self.assertIn('errors', report)
        self.assertIn('warnings', report)
        self.assertIn('recommendations', report)
        
        # 测试Android兼容性检查
        compatibility = self.config_validator.check_android_compatibility()
        self.assertIn('status', compatibility)
        self.assertIn('issues', compatibility)
        self.assertIn('recommendations', compatibility)
        
    def test_performance_monitor(self):
        """测试性能监控器"""
        # 开始性能监控
        self.performance_monitor.start_monitoring()
        
        # 模拟一些操作
        time.sleep(0.1)
        
        # 获取性能指标
        metrics = self.performance_monitor.get_metrics()
        self.assertIsInstance(metrics, dict)
        
        # 停止监控
        self.performance_monitor.stop_monitoring()
        
        # 测试性能报告
        report = self.performance_monitor.generate_report()
        self.assertIn('metrics', report)
        self.assertIn('summary', report)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试基本异常
        with self.assertRaises(MobileLLMError):
            raise MobileLLMError("Test error")
            
        # 测试配置错误
        with self.assertRaises(InvalidConfigError):
            raise InvalidConfigError("Test config error", "test_key")
            
    def test_mtk_config_validation(self):
        """测试MTK配置验证"""
        validation = self.config_validator.validate_mtk_config()
        self.assertTrue(validation['status'])
        self.assertIsInstance(validation['issues'], list)
        self.assertIsInstance(validation['recommendations'], list)
        
    def test_quantization_config_validation(self):
        """测试量化配置验证"""
        validation = self.config_validator.validate_quantization_config()
        self.assertTrue(validation['status'])
        self.assertIsInstance(validation['issues'], list)
        self.assertIsInstance(validation['recommendations'], list)

if __name__ == '__main__':
    unittest.main()