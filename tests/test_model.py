# 文件路径: tests/test_model.py
# 新建文件

import unittest
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_config import ModelConfig
from script.model.tts_model_adapter import TTSModelAdapter

class TestTTSModel(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_model_loading(self):
        """测试模型加载"""
        try:
            model = TTSModelAdapter(self.config)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Model loading failed: {str(e)}")
    
    def test_inference(self):
        """测试模型推理"""
        model = TTSModelAdapter(self.config)
        test_text = "这是一个测试文本"
        
        try:
            output = model.generate_speech(test_text)
            self.assertIsNotNone(output)
            self.assertTrue(isinstance(output, torch.Tensor))
        except Exception as e:
            self.fail(f"Inference failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()