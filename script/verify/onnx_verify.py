# 文件路径: script/verify/onnx_verify.py

import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
import time
from transformers import AutoTokenizer
import onnx  # 添加 onnx 导入


logger = logging.getLogger(__name__)

class ONNXModelVerifier:
    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path.parent
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
    def analyze_model_structure(self):
        """分析ONNX模型结构"""
        try:
            logger.info("Analyzing ONNX model structure...")
            model = onnx.load(str(self.model_path))
            
            # 查看所有节点
            for node in model.graph.node:
                if node.op_type == 'Slice':
                    logger.info(f"Found Slice node: {node.name}")
                    logger.info(f"Inputs: {node.input}")
                    logger.info(f"Outputs: {node.output}")
                    
                    # 查找相关的输入值信息
                    for init in model.graph.initializer:
                        if init.name in node.input:
                            logger.info(f"Initializer {init.name}: shape={init.dims}, data_type={init.data_type}")
            
            return True
        except Exception as e:
            logger.error(f"Model analysis failed: {e}")
            return False
    
    def load_session(self):
        """加载ONNX会话"""
        try:
            logger.info("Creating ONNX Runtime session...")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                session_options,
                providers=['CPUExecutionProvider']
            )
            
            # 获取所有输入信息
            self.input_details = self.session.get_inputs()
            logger.info("Model inputs:")
            for input_detail in self.input_details:
                logger.info(f"  Name: {input_detail.name}, Shape: {input_detail.shape}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX session: {e}")
            return False
    
    def verify_basic_inference(self):
        """验证基本推理功能"""
        try:
            logger.info("Testing basic inference...")
            
            # 准备所有输入
            inputs = {}
            for input_detail in self.input_details:
                logger.info(f"Preparing input for: {input_detail.name}")
                
                if input_detail.name == 'input_0':
                    # 主输入使用随机token IDs
                    dummy_input = np.random.randint(0, 100, (1, 32), dtype=np.int64)
                    inputs[input_detail.name] = dummy_input
                    logger.info(f"Input shape for {input_detail.name}: {dummy_input.shape}")
                elif input_detail.name == 'onnx::Neg_1':
                    # Slice 操作需要1维数组
                    dummy_neg = np.zeros(1, dtype=np.int64)  # 单个元素的1维数组
                    inputs[input_detail.name] = dummy_neg
                    logger.info(f"Input shape for {input_detail.name}: {dummy_neg.shape}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行推理
            outputs = self.session.run(None, inputs)
            
            # 计算耗时
            inference_time = time.time() - start_time
            logger.info(f"Basic inference successful! Time taken: {inference_time*1000:.2f}ms")
            logger.info(f"Output shape: {outputs[0].shape}")
            
            return True
        except Exception as e:
            logger.error(f"Basic inference failed: {e}")
            logger.error(f"Available input names: {[input.name for input in self.session.get_inputs()]}")
            if hasattr(e, '__cause__'):
                logger.error(f"Cause: {e.__cause__}")
            return False
    
    def verify_with_real_input(self):
        """使用真实文本输入验证"""
        try:
            logger.info("Testing with real text input...")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
            
            # 准备测试文本
            test_text = "你好，请问今天天气如何？"
            logger.info(f"Test input text: {test_text}")
            
            # 对文本进行编码
            encoded = tokenizer(test_text, return_tensors="np")
            
            # 准备所有输入
            inputs = {}
            for input_detail in self.input_details:
                if input_detail.name == 'input_0':
                    inputs[input_detail.name] = encoded['input_ids']
                elif input_detail.name == 'onnx::Neg_1':
                    inputs[input_detail.name] = np.array([], dtype=np.int64)
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行推理
            outputs = self.session.run(None, inputs)
            
            # 计算耗时
            inference_time = time.time() - start_time
            
            # 解码输出
            output_text = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
            
            logger.info(f"Real text inference successful! Time taken: {inference_time*1000:.2f}ms")
            logger.info(f"Model output: {output_text}")
            
            return True
        except Exception as e:
            logger.error(f"Real text inference failed: {e}")
            if hasattr(e, '__cause__'):
                logger.error(f"Cause: {e.__cause__}")
            return False
    
    def run_performance_test(self, num_iterations=10):
        """运行性能测试"""
        try:
            logger.info(f"Running performance test with {num_iterations} iterations...")
            
            latencies = []
            dummy_input = np.random.randint(0, 100, (1, 32), dtype=np.int64)
            
            # 预热
            logger.info("Warming up...")
            for _ in range(3):
                self.session.run(None, {self.input_name: dummy_input})
            
            # 性能测试
            logger.info("Running inference tests...")
            for i in range(num_iterations):
                start_time = time.time()
                self.session.run(None, {self.input_name: dummy_input})
                latencies.append((time.time() - start_time) * 1000)  # 转换为毫秒
            
            # 计算统计信息
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            logger.info(f"Performance test results:")
            logger.info(f"Average latency: {avg_latency:.2f}ms")
            logger.info(f"Min latency: {min_latency:.2f}ms")
            logger.info(f"Max latency: {max_latency:.2f}ms")
            
            return True
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False

def verify_onnx_model(model_path: str, tokenizer_path: str = None):
    """完整的验证流程"""
    try:
        verifier = ONNXModelVerifier(model_path, tokenizer_path)
        
        # 1. 加载会话
        if not verifier.load_session():
            return False
        
        # 2. 基本推理测试
        if not verifier.verify_basic_inference():
            return False
        
        # 3. 真实文本测试
        if not verifier.verify_with_real_input():
            return False
        
        # 4. 性能测试
        if not verifier.run_performance_test():
            return False
        
        logger.info("All verification tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    # 模型路径
    model_path = project_root / "models" / "android" / "model.onnx"
    tokenizer_path = project_root / "models" / "original"
    
    # 运行验证
    success = verify_onnx_model(str(model_path), str(tokenizer_path))
    sys.exit(0 if success else 1)