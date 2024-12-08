import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
import time
from transformers import AutoTokenizer
import onnx
from onnx import numpy_helper
import traceback

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
            logger.info("创建ONNX Runtime会话...")
            
            # 分析模型结构
            self.analyze_model_inputs()
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                session_options,
                providers=['CPUExecutionProvider']
            )
            
            # 获取输入详情
            self.input_details = self.session.get_inputs()
            logger.info("模型输入信息:")
            for input_detail in self.input_details:
                logger.info(f"  名称: {input_detail.name}, 形状: {input_detail.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载ONNX会话失败: {e}")
            logger.error(traceback.format_exc())
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

    def analyze_model_inputs(self):
        """分析模型的输入节点"""
        try:
            logger.info("分析模型输入结构...")
            model = onnx.load(str(self.model_path))
            
            # 检查所有输入
            input_names = []
            for input in model.graph.input:
                logger.info(f"发现输入节点: {input.name}")
                input_names.append(input.name)
                
                # 分析输入的形状信息
                shape_info = []
                for dim in input.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape_info.append(dim.dim_param)
                    else:
                        shape_info.append(dim.dim_value)
                logger.info(f"  形状信息: {shape_info}")
            
            # 分析Slice节点的输入
            for node in model.graph.node:
                if node.op_type == 'Slice':
                    logger.info(f"\nSlice节点分析:")
                    logger.info(f"节点名称: {node.name}")
                    logger.info(f"输入列表: {node.input}")
                    
                    # 检查initializer
                    for init in model.graph.initializer:
                        if init.name in node.input:
                            logger.info(f"找到initializer: {init.name}")
                            logger.info(f"  形状: {[d for d in init.dims]}")
            
            return input_names
        except Exception as e:
            logger.error(f"分析模型输入时出错: {e}")
            return []

    def verify_basic_inference(self):
        """验证基本推理功能"""
        try:
            logger.info("开始基本推理测试...")
            
            # 准备输入数据
            inputs = {}
            
            # 检查并记录所有需要的输入
            required_inputs = set(input.name for input in self.input_details)
            logger.info(f"模型需要的输入: {required_inputs}")
            
            for input_detail in self.input_details:
                logger.info(f"处理输入: {input_detail.name}")
                
                if input_detail.name == 'input_0':
                    # 为主要输入创建测试张量
                    dummy_input = np.random.randint(0, 100, (1, 32), dtype=np.int64)
                    inputs[input_detail.name] = dummy_input
                    logger.info(f"创建了input_0张量: shape={dummy_input.shape}, dtype={dummy_input.dtype}")
                    
                elif input_detail.name == 'onnx::Neg_1':
                    # 严格确保starts是一维数组
                    starts = np.array([0], dtype=np.int64)
                    starts = np.ascontiguousarray(starts)  # 确保数组连续存储
                    
                    # 验证starts数组的属性
                    logger.info(f"starts数组信息:")
                    logger.info(f"  shape: {starts.shape}")
                    logger.info(f"  dtype: {starts.dtype}")
                    logger.info(f"  ndim: {starts.ndim}")
                    logger.info(f"  strides: {starts.strides}")
                    logger.info(f"  flags: {starts.flags}")
                    logger.info(f"  数据: {starts}")
                    
                    inputs[input_detail.name] = starts

            # 打印最终的输入配置
            logger.info("\n最终输入配置:")
            for name, value in inputs.items():
                logger.info(f"  {name}:")
                logger.info(f"    shape: {value.shape}")
                logger.info(f"    dtype: {value.dtype}")
                logger.info(f"    ndim: {value.ndim}")
                logger.info(f"    值: {value}")
            
            # 运行推理
            logger.info("\n执行推理...")
            try:
                outputs = self.session.run(None, inputs)
                logger.info("推理成功完成!")
                
                # 打印输出信息
                for i, output in enumerate(outputs):
                    logger.info(f"输出 {i} 形状: {output.shape}")
                
                return True
                
            except Exception as e:
                logger.error(f"推理过程中出错: {e}")
                logger.error(traceback.format_exc())
                
                # 检查错误是否与输入形状相关
                if "Starts must be a 1-D array" in str(e):
                    logger.error("\n输入形状错误分析:")
                    for name, value in inputs.items():
                        logger.error(f"  {name} 的形状信息:")
                        logger.error(f"    shape: {value.shape}")
                        logger.error(f"    ndim: {value.ndim}")
                        logger.error(f"    dtype: {value.dtype}")
                        logger.error(f"    strides: {value.strides}")
                
                return False
                
        except Exception as e:
            logger.error(f"基本推理测试失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def fix_slice_inputs(self, model_path):
        """修复Slice节点的输入"""
        try:
            # 加载原始模型
            model = onnx.load(model_path)
            logger.info("正在修复Slice节点输入...")
            
            # 分析Slice节点的输入
            for node in model.graph.node:
                if node.op_type == 'Slice':
                    logger.info(f"找到Slice节点: {node.name}")
                    logger.info(f"当前输入: {node.input}")
                    
                    # 保持第一个输入不变，修改starts输入
                    if len(node.input) > 1:
                        # 创建新的starts initializer
                        starts_name = node.input[1]
                        starts_value = np.array([0], dtype=np.int64)
                        
                        # 检查并移除旧的initializer
                        model.graph.initializer[:] = [x for x in model.graph.initializer 
                                                    if x.name != starts_name]
                        
                        # 添加新的initializer
                        new_starts = onnx.helper.make_tensor(
                            name=starts_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=starts_value.tolist()
                        )
                        model.graph.initializer.append(new_starts)
                        logger.info(f"更新了starts输入: {starts_name}")
                        logger.info(f"  形状: {[1]}")
                        logger.info(f"  值: {starts_value}")
            
            # 保存修复后的模型
            fixed_model_path = model_path.replace('.onnx', '_fixed.onnx')
            onnx.save(model, fixed_model_path)
            logger.info(f"修复后的模型已保存到: {fixed_model_path}")
            
            return fixed_model_path
        
        except Exception as e:
            logger.error(f"修复Slice输入时出错: {e}")
            logger.error(traceback.format_exc())
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