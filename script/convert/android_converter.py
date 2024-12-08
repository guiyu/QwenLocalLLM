# 文件路径: script/convert/android_converter.py

import logging
import torch
import numpy as np
from pathlib import Path
import traceback
from transformers import AutoModelForCausalLM
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
from onnx import helper, TensorProto

logger = logging.getLogger(__name__)

class AndroidModelConverter:
    def __init__(self, model_path, output_dir="./models/android"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_onnx_model_proto(self, graph_def, opset_imports):
        """创建ONNX模型原型"""
        model_def = helper.make_model(
            graph_def,
            producer_name='pytorch',
            opset_imports=opset_imports
        )
        model_def.ir_version = 7  # 使用稳定版本的IR
        return model_def

    def convert_model(self):
        """执行模型转换"""
        try:
            logger.info("正在加载模型...")
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model.eval()

            logger.info("准备模型转换...")
            
            # 创建示例输入
            dummy_input = torch.randint(100, (1, 32), dtype=torch.long)
            logger.info(f"创建示例输入: shape={dummy_input.shape}, dtype={dummy_input.dtype}")
            
            # 包装模型
            class ModelWrapper(torch.nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.model = base_model

                def forward(self, input_ids):
                    outputs = self.model(input_ids)
                    return outputs.logits.float()

            wrapped_model = ModelWrapper(model)

            # 导出ONNX
            logger.info("开始导出ONNX模型...")
            initial_path = self.output_dir / "model_initial.onnx"
            
            # 设置导出配置
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'}
            }
            
            # 导出前检查
            logger.info("验证模型前向传播...")
            with torch.no_grad():
                test_output = wrapped_model(dummy_input)
                logger.info(f"测试输出形状: {test_output.shape}")

            # 导出模型
            torch.onnx.export(
                wrapped_model,
                (dummy_input,),
                str(initial_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=True  # 启用详细日志
            )

            logger.info("修复ONNX模型...")
            try:
                # 加载和验证初始模型
                logger.info("加载初始模型...")
                initial_model = onnx.load(str(initial_path))
                logger.info(f"初始模型IR版本: {initial_model.ir_version}")
                
                # 设置IR版本
                initial_model.ir_version = 7
                logger.info(f"设置IR版本后的值: {initial_model.ir_version}")
                
                # 保存模型
                fixed_path = self.output_dir / "model.onnx"
                logger.info(f"保存模型到: {fixed_path}")
                
                try:
                    # 保存模型
                    logger.info("尝试使用onnx.save保存...")
                    onnx.save(initial_model, str(fixed_path), save_as_external_data=True, all_tensors_to_one_file=True, location="weights.pb")
                    logger.info("模型保存成功")
                    
                    # 验证保存的文件
                    logger.info("验证保存的文件...")
                    if fixed_path.exists():
                        file_size = fixed_path.stat().st_size
                        logger.info(f"主文件大小: {file_size} 字节")
                        weights_path = fixed_path.parent / "weights.pb"
                        if weights_path.exists():
                            logger.info(f"权重文件大小: {weights_path.stat().st_size} 字节")
                
                except Exception as save_error:
                    logger.error(f"保存模型时出错: {save_error}")
                    logger.error(traceback.format_exc())
                    raise
                
                # 使用文件路径进行验证
                logger.info("验证最终模型...")
                onnx.checker.check_model(str(fixed_path))  # 使用文件路径而不是模型对象
                logger.info("模型验证通过")
                
            except Exception as e:
                logger.error(f"ONNX模型处理失败: {str(e)}")
                logger.error(traceback.format_exc())
                if 'initial_model' in locals():
                    logger.error(f"失败时的IR版本: {getattr(initial_model, 'ir_version', 'Not found')}")
                raise
            
            # 设置IR版本
            initial_model.ir_version = 7  # 使用正确的变量名 initial_model
            
            # 设置模型版本和文档
            initial_model.model_version = 1
            initial_model.doc_string = "Converted from PyTorch"
            
            # 保存修复后的模型
            fixed_path = self.output_dir / "model.onnx"
            with open(fixed_path, 'wb') as f:
                f.write(initial_model.SerializeToString())

            # 检查保存的模型
            logger.info("检查修复后的模型...")
            # with open(fixed_path, 'rb') as f:
            #     fixed_model = onnx.load_from_string(f.read())
                        
            # 运行模型检查器
            logger.info("运行ONNX模型检查器...")
            # onnx.checker.check_model(fixed_model)

            # 优化模型
            logger.info("优化模型...")
            import onnxruntime as ort
            
            # 配置优化选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1
            
            # 设置优化后的模型路径
            optimized_path = self.output_dir / "model_optimized.onnx"
            sess_options.optimized_model_filepath = str(optimized_path)
            
            # 创建优化会话
            _ = ort.InferenceSession(
                str(fixed_path),
                sess_options,
                providers=['CPUExecutionProvider']
            )

            # 量化模型
            logger.info("量化模型...")
            quantized_path = self.output_dir / "model_quantized.onnx"
            logger.info("量化模型...")
            from onnx import TensorProto
            extra_options = {
                'DefaultTensorType': TensorProto.FLOAT  # 设置默认张量类型为FLOAT
            }

            quantize_dynamic(
                model_input=str(optimized_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
                per_channel=False,
                extra_options=extra_options  # 添加额外选项
            )
            # 验证最终模型
            logger.info("验证最终模型...")
            success = self.verify_converted_model(str(quantized_path))
            if not success:
                raise Exception("最终模型验证失败")

            logger.info(f"模型转换完成，文件保存在: {self.output_dir}")
            return str(quantized_path)

        except Exception as e:
            logger.error(f"模型转换失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def verify_converted_model(self, model_path):
        """验证转换后的模型"""
        try:
            logger.info(f"验证模型: {model_path}")
            
            # 检查模型格式
            logger.info("检查模型格式...")
            onnx_model = onnx.load(model_path)
            logger.info(f"IR版本: {onnx_model.ir_version}")
            logger.info(f"Opset版本: {[opset.version for opset in onnx_model.opset_import]}")
            
            # 创建推理会话
            import onnxruntime as ort
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            
            logger.info("创建推理会话...")
            session = ort.InferenceSession(
                model_path,
                session_options,
                providers=['CPUExecutionProvider']
            )

            # 输入信息
            input_details = session.get_inputs()
            logger.info("模型输入信息：")
            for input in input_details:
                logger.info(f"  名称: {input.name}")
                logger.info(f"  形状: {input.shape}")
                logger.info(f"  类型: {input.type}")

            # 测试推理
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randint(0, 100, (1, 32), dtype=np.int64)
            
            logger.info("执行推理...")
            outputs = session.run(None, {input_name: dummy_input})
            
            logger.info("推理成功!")
            for i, output in enumerate(outputs):
                logger.info(f"输出 {i} 形状: {output.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型验证失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def convert_for_android(model_path, output_dir, config, basic_mode=False):
    """主转换函数"""
    try:
        logger.info("开始Android模型转换...")
        
        converter = AndroidModelConverter(model_path, output_dir)
        converted_path = converter.convert_model()
        
        logger.info(f"转换完成，模型保存在: {converted_path}")
        return converted_path
        
    except Exception as e:
        logger.error(f"转换过程中出错: {str(e)}")
        raise