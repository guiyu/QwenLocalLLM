# 文件路径: script/convert/android_converter.py
# 新建文件

import torch
import onnx
from pathlib import Path
import logging
import shutil
import json
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class AndroidModelConverter:
    def __init__(self, model_path, output_dir="./models/android"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_model(self):
        """执行模型转换"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info("Loading model and tokenizer...")
            # 使用半精度加载模型以减少内存使用
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,  # 使用 FP16
                low_cpu_mem_usage=True
            )
            model.eval()
            
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            logger.info("Converting model to ONNX format...")
            onnx_path = self.output_dir / "model.onnx"
            
            # 配置ONNX导出选项
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            }
            
            # 使用较小的序列长度进行导出
            dummy_input = torch.randint(100, (1, 32), dtype=torch.long)  # 减小序列长度
            
            logger.info("Starting ONNX export (this may take a while)...")
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )
            
            # 使用文件路径验证模型
            logger.info("Verifying exported ONNX model...")
            import onnx
            onnx.checker.check_model(str(onnx_path))
            
            # 检查文件大小
            file_size_gb = onnx_path.stat().st_size / (1024 * 1024 * 1024)
            logger.info(f"Exported ONNX model size: {file_size_gb:.2f} GB")
            
            if file_size_gb > 2:
                logger.warning("Model is quite large. Consider further optimization.")
                
                # 尝试进行优化
                logger.info("Applying post-export optimizations...")
                import onnxruntime as ort
                
                # 配置会话选项
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.optimized_model_filepath = str(self.output_dir / "model_optimized.onnx")
                
                # 创建优化会话
                _ = ort.InferenceSession(
                    str(onnx_path),
                    sess_options,
                    providers=['CPUExecutionProvider']
                )
                
                # 使用优化后的模型路径
                onnx_path = Path(sess_options.optimized_model_filepath)
                
                # 再次检查文件大小
                optimized_size_gb = onnx_path.stat().st_size / (1024 * 1024 * 1024)
                logger.info(f"Optimized model size: {optimized_size_gb:.2f} GB")
            
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"Error during model conversion: {e}")
            raise
    
    def optimize_for_mobile(self, onnx_model):
        """优化ONNX模型以适应移动端"""
        logger.info("Optimizing ONNX model for mobile")
        
        # 图优化选项
        from onnxruntime import GraphOptimizationLevel, SessionOptions
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        options.optimize_for_mobile = True
        
        # 执行图优化
        import onnxruntime as ort
        session = ort.InferenceSession(
            str(self.model_path), 
            options,
            providers=['CPUExecutionProvider']
        )
        
        # 保存优化后的模型
        optimized_path = self.output_dir / "model_optimized.onnx"
        onnx.save(onnx_model, str(optimized_path))
        
        return str(optimized_path)
    
    def convert_to_lite(self, optimized_path):
        """转换为TFLite格式（可选）"""
        try:
            import tensorflow as tf
            from onnx_tf.backend import prepare
            
            # ONNX to TensorFlow
            logger.info("Converting ONNX to TensorFlow")
            tf_model = prepare(onnx.load(optimized_path))
            
            # TensorFlow to TFLite
            logger.info("Converting TensorFlow to TFLite")
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model.keras_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
            
            # 保存TFLite模型
            tflite_path = self.output_dir / "model.tflite"
            tflite_path.write_bytes(tflite_model)
            
            return str(tflite_path)
        except ImportError:
            logger.warning("TensorFlow not installed, skipping TFLite conversion")
            return None
    
    def prepare_android_assets(self):
        """准备Android资产文件"""
        logger.info("Preparing Android assets")
        
        # 创建assets目录
        assets_dir = self.output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # 复制模型文件
        shutil.copy2(self.model_path, assets_dir / "model_quantized.onnx")
        
        # 创建模型配置文件
        config = {
            "model_version": "1.0",
            "input_shape": [1, 64],
            "input_name": "input",
            "output_name": "output",
            "quantization_bits": 8
        }
        
        with open(assets_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        return str(assets_dir)
    
    def generate_android_helper(self):
        """生成Android辅助类"""
        logger.info("Generating Android helper classes")
        
        java_code = """
package com.example.qwentts;

import android.content.Context;
import android.util.Log;
import ai.onnxruntime.*;

public class QwenTTSModel {
    private static final String TAG = "QwenTTSModel";
    private OrtSession session;
    private OrtEnvironment env;
    
    public QwenTTSModel(Context context) {
        try {
            // 初始化ONNX Runtime
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            
            // 从assets加载模型
            byte[] modelBytes = Utils.loadModelFromAssets(context, "model_quantized.onnx");
            session = env.createSession(modelBytes, sessionOptions);
        } catch (Exception e) {
            Log.e(TAG, "Error initializing model", e);
        }
    }
    
    public float[] generateSpeech(String text) {
        try {
            // 处理输入文本
            long[] inputShape = {1, text.length()};
            OnnxTensor inputTensor = OnnxTensor.createTensor(
                env,
                Utils.textToInputArray(text),
                inputShape
            );
            
            // 运行推理
            OrtSession.Result result = session.run(
                Collections.singletonMap("input", inputTensor)
            );
            
            // 处理输出
            return Utils.processOutput(result);
        } catch (Exception e) {
            Log.e(TAG, "Error generating speech", e);
            return null;
        }
    }
    
    public void close() {
        try {
            if (session != null) session.close();
            if (env != null) env.close();
        } catch (Exception e) {
            Log.e(TAG, "Error closing model", e);
        }
    }
}
"""
        
        # 保存Java代码
        java_dir = self.output_dir / "java" / "com" / "example" / "qwentts"
        java_dir.mkdir(parents=True, exist_ok=True)
        
        with open(java_dir / "QwenTTSModel.java", "w") as f:
            f.write(java_code)
        
        return str(java_dir)

# 文件路径: script/convert/android_converter.py
# 修改现有文件，添加基础模式选项

def convert_for_android(model_path, output_dir, config, basic_mode=False):
    try:
        logger.info("Starting Android model conversion...")
        
        # 检查并转换路径
        model_path = Path(model_path).resolve()
        output_dir = Path(output_dir).resolve()
        
        # 检查源目录
        logger.info(f"Checking model directory: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # 检查目录内容
        model_files = list(model_path.glob('*'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_path}")
            
        logger.info(f"Found {len(model_files)} files in model directory")
        for file in model_files:
            logger.info(f"Found model file: {file.name}")
            # 验证文件可读性
            try:
                with open(file, 'rb') as f:
                    # 只读取一小部分来验证可访问性
                    f.read(1024)
                logger.info(f"Successfully verified read access for {file.name}")
            except PermissionError:
                logger.error(f"Permission denied when reading {file}")
                raise
            except Exception as e:
                logger.error(f"Error accessing {file}: {str(e)}")
                raise

        # 检查输出目录
        logger.info(f"Checking output directory: {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / 'test_write.txt'
            test_file.write_text('test')
            test_file.unlink()  # 删除测试文件
            logger.info("Successfully verified write access to output directory")
        except PermissionError:
            logger.error(f"Permission denied for output directory: {output_dir}")
            raise
        except Exception as e:
            logger.error(f"Error accessing output directory: {str(e)}")
            raise
        
        converter = AndroidModelConverter(model_path, output_dir)
        onnx_path = converter.convert_model()
        
        logger.info(f"Model converted successfully to: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise