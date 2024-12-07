import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
import time
from transformers import AutoTokenizer
import onnx
from onnx import numpy_helper

logger = logging.getLogger(__name__)

class ONNXModelVerifier:
    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path.parent
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def analyze_onnx_model(self):
        """分析ONNX模型结构"""
        model = onnx.load(str(self.model_path))
        logger.info("Analyzing ONNX model structure...")
        
        # 1. 分析所有initializer
        initializers = {init.name: init for init in model.graph.initializer}
        for name, init in initializers.items():
            logger.info(f"\nInitializer: {name}")
            logger.info(f"  Shape: {list(init.dims)}")
            logger.info(f"  Data type: {init.data_type}")
            if len(init.dims) == 0:
                logger.info("  WARNING: Scalar initializer found!")

        # 2. 分析所有Slice节点
        for node in model.graph.node:
            if node.op_type == 'Slice':
                logger.info(f"\nSlice node: {node.name}")
                logger.info("  Inputs:")
                for idx, input_name in enumerate(node.input):
                    logger.info(f"    {idx}: {input_name}")
                    if input_name in initializers:
                        init = initializers[input_name]
                        logger.info(f"      (Initializer) Shape: {list(init.dims)}, Type: {init.data_type}")
                    else:
                        # 查找产生这个输入的节点
                        source_node = None
                        for n in model.graph.node:
                            if input_name in n.output:
                                source_node = n
                                break
                        if source_node:
                            logger.info(f"      Source node: {source_node.op_type} ({source_node.name})")

        return model

    def modify_model_if_needed(self, model):
        """修改ONNX模型的Slice节点输入"""
        modified = False
        
        # 找到需要修改的Slice节点
        for node in model.graph.node:
            if node.op_type == 'Slice' and len(node.input) == 5:
                # 检查是否需要修改
                for input_node in model.graph.node:
                    if input_node.op_type == 'Unsqueeze' and input_node.output[0] in node.input:
                        # 修改Unsqueeze节点的attributes
                        for attr in input_node.attribute:
                            if attr.name == 'axes':
                                original_axes = list(attr.ints)
                                if original_axes != [0]:
                                    attr.ints[:] = [0]  # 修改为只在第0维添加维度
                                    modified = True
                                    logger.info(f"Modified Unsqueeze node axes from {original_axes} to [0]")

        if modified:
            modified_path = str(self.model_path).replace('.onnx', '_modified.onnx')
            onnx.save(model, modified_path)
            logger.info(f"Saved modified model to {modified_path}")
            return modified_path
        
        return str(self.model_path)

    def load_session(self):
        """加载ONNX会话"""
        try:
            logger.info("Creating ONNX Runtime session...")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 分析和修改模型
            model = self.analyze_onnx_model()
            model_path = self.modify_model_if_needed(model)
            
            self.session = ort.InferenceSession(
                model_path,
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

    def preprocess_inputs(self, inputs):
        """预处理输入数据"""
        processed = {}
        for name, value in inputs.items():
            if name == 'onnx::Neg_1':
                # 确保是1维数组
                value = np.asarray(value, dtype=np.int64)
                if value.ndim != 1:
                    value = value.reshape(-1)
                value = np.ascontiguousarray(value)
                logger.info(f"Preprocessed {name}:")
                logger.info(f"  Shape: {value.shape}")
                logger.info(f"  Dtype: {value.dtype}")
                logger.info(f"  Strides: {value.strides}")
                logger.info(f"  Flags: {value.flags}")
            processed[name] = value
        return processed

    def verify_basic_inference(self):
        """验证基本推理功能"""
        try:
            logger.info("Testing basic inference...")
            model = onnx.load(str(self.model_path))
            
            # 分析Constant节点
            logger.info("Analyzing Constants...")
            constants = {}
            for node in model.graph.node:
                if node.op_type == 'Constant':
                    logger.info(f"Found Constant node: {node.name}")
                    for attr in node.attribute:
                        if attr.name == 'value':
                            tensor = numpy_helper.to_array(attr.t)
                            logger.info(f"  Shape: {tensor.shape}")
                            logger.info(f"  Value: {tensor}")
                            constants[node.output[0]] = tensor

            # 分析有问题的Slice节点
            logger.info("\nAnalyzing problematic Slice node...")
            for node in model.graph.node:
                if node.op_type == 'Slice' and node.name == '/Slice':
                    logger.info(f"Found target Slice node inputs:")
                    for i, input_name in enumerate(node.input):
                        logger.info(f"Input {i}: {input_name}")
                        if input_name in constants:
                            logger.info(f"  Constant value: {constants[input_name]}")

            # 准备输入
            inputs = {}
            for input_detail in self.input_details:
                logger.info(f"\nPreparing input for: {input_detail.name}")
                
                if input_detail.name == 'input_0':
                    dummy_input = np.random.randint(0, 100, (1, 32), dtype=np.int64)
                    inputs[input_detail.name] = dummy_input
                    logger.info(f"Input shape: {dummy_input.shape}")
                elif input_detail.name == 'onnx::Neg_1':
                    # 创建完整的Slice参数
                    starts = np.zeros(1, dtype=np.int64)  # starts
                    ends = np.array([1], dtype=np.int64)  # ends
                    axes = np.array([0], dtype=np.int64)  # axes
                    steps = np.array([1], dtype=np.int64)  # steps
                    
                    inputs[input_detail.name] = starts
                    inputs['slice_ends'] = ends
                    inputs['slice_axes'] = axes
                    inputs['slice_steps'] = steps
                    
                    logger.info(f"Slice parameters:")
                    logger.info(f"  starts: shape={starts.shape}, value={starts}")
                    logger.info(f"  ends: shape={ends.shape}, value={ends}")
                    logger.info(f"  axes: shape={axes.shape}, value={axes}")
                    logger.info(f"  steps: shape={steps.shape}, value={steps}")

            # 运行推理
            logger.info("\nAttempting inference...")
            outputs = self.session.run(None, inputs)
            
            logger.info("Inference successful!")
            for i, output in enumerate(outputs):
                logger.info(f"Output {i} shape: {output.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Basic inference failed: {e}")
            return False
        
    def fix_slice_inputs(self, model_path):
        """修复Slice节点的输入"""
        try:
            # 加载原始模型
            model = onnx.load(model_path)
            
            logger.info("Fixing Slice node inputs...")
            
            # 找到有问题的Slice节点
            slice_node = None
            for node in model.graph.node:
                if node.op_type == 'Slice' and node.name == '/Slice':
                    slice_node = node
                    break
            
            if slice_node:
                # 确保所有输入是正确的一维数组
                input_tensors = {
                    'starts': np.array([0], dtype=np.int64),
                    'ends': np.array([1], dtype=np.int64),
                    'axes': np.array([0], dtype=np.int64),
                    'steps': np.array([1], dtype=np.int64)
                }
                
                # 修改或添加对应的initializers
                for i, (name, tensor) in enumerate(input_tensors.items()):
                    if i < len(slice_node.input):
                        init_name = slice_node.input[i]
                    else:
                        init_name = f"fixed_{name}"
                        slice_node.input.append(init_name)
                    
                    # 创建新的initializer
                    initializer = numpy_helper.from_array(tensor, name=init_name)
                    
                    # 删除旧的initializer（如果存在）
                    model.graph.initializer[:] = [x for x in model.graph.initializer if x.name != init_name]
                    
                    # 添加新的initializer
                    model.graph.initializer.append(initializer)
                
                # 保存修改后的模型
                fixed_model_path = model_path.replace('.onnx', '_fixed.onnx')
                onnx.save(model, fixed_model_path)
                logger.info(f"Saved fixed model to {fixed_model_path}")
                
                return fixed_model_path
        
        except Exception as e:
            logger.error(f"Failed to fix Slice inputs: {e}")
            return model_path

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
            
            # 准备输入
            inputs = {}
            for input_detail in self.input_details:
                if input_detail.name == 'input_0':
                    inputs[input_detail.name] = encoded['input_ids']
                elif input_detail.name == 'onnx::Neg_1':
                    inputs[input_detail.name] = np.array([0], dtype=np.int64)
            
            # 预处理输入
            inputs = self.preprocess_inputs(inputs)
            
            # 运行推理
            outputs = self.session.run(None, inputs)
            
            # 解码输出
            output_text = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
            
            logger.info(f"Real text inference successful!")
            logger.info(f"Model output: {output_text}")
            
            return True
        except Exception as e:
            logger.error(f"Real text inference failed: {e}")
            if hasattr(e, '__cause__'):
                logger.error(f"Cause: {e.__cause__}")
            return False

def verify_onnx_model(model_path: str, tokenizer_path: str = None):
    """完整的验证流程"""
    try:
        verifier = ONNXModelVerifier(model_path, tokenizer_path)
        
        # 1. 分析并修复模型
        fixed_model_path = verifier.fix_slice_inputs(model_path)
        
        # 2. 使用修复后的模型创建验证器
        verifier = ONNXModelVerifier(fixed_model_path, tokenizer_path)
        
        # 3. 加载会话
        if not verifier.load_session():
            return False
        
        # 4. 基本推理测试
        if not verifier.verify_basic_inference():
            return False
        
        # 5. 真实文本测试
        if not verifier.verify_with_real_input():
            return False
        
        logger.info("All verification tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    # 模型路径
    model_path = project_root / "models" / "android" / "model.onnx"
    tokenizer_path = project_root / "models" / "original"
    
    # 运行验证
    success = verify_onnx_model(str(model_path), str(tokenizer_path))
    import sys
    sys.exit(0 if success else 1)