import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic
import os
import json
import onnx
import onnxruntime as ort

class ModelOptimizer:
    def __init__(self, model_path="./qwen_small_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        print("Initializing ModelOptimizer...")
    
    def load_model(self):
        """加载原始模型"""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        return self
    
    def apply_pruning(self, pruning_amount=0.3):
        """应用结构化剪枝"""
        print(f"Applying pruning with amount {pruning_amount}...")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=pruning_amount
                )
                prune.remove(module, 'weight')
        return self
    
    def optimize_attention(self):
        """优化注意力机制"""
        print("Optimizing attention mechanism...")
        # 实现注意力机制优化
        # 例如：调整attention pattern, 实现KV cache等
        return self
    
    def quantize_model(self):
        """量化模型"""
        print("Quantizing model...")
        self.model.eval()
        # 动态量化
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # 对Linear层进行量化
            dtype=torch.qint8
        )
        return self
    
    def export_to_onnx(self, output_path="mobile_optimized.onnx"):
        """导出为ONNX格式"""
        print("Exporting to ONNX...")
        
        # 准备示例输入
        dummy_input = torch.randint(100, (1, 64))  # batch_size=1, sequence_length=64
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            },
            opset_version=13,
            do_constant_folding=True
        )
        
        # 优化ONNX模型
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic(
            output_path,
            output_path.replace('.onnx', '_quantized.onnx'),
            weight_type=torch.qint8
        )
        return self
    
    def verify_performance(self, test_text="你好，请问今天天气如何？"):
        """验证模型性能"""
        print("\nVerifying model performance...")
        
        # 测试推理性能
        inputs = self.tokenizer(test_text, return_tensors="pt")
        
        # 测试延迟
        import time
        latencies = []
        
        for _ in range(5):
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7
                )
            latencies.append(time.time() - start_time)
        
        avg_latency = sum(latencies) / len(latencies)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Average inference latency: {avg_latency*1000:.2f}ms")
        print(f"Test response: {response}")
        
        # 保存性能指标
        metrics = {
            "avg_latency_ms": avg_latency * 1000,
            "model_size_mb": os.path.getsize("mobile_optimized_quantized.onnx") / (1024 * 1024)
        }
        
        with open("model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

def optimize_model():
    optimizer = ModelOptimizer()
    
    # 执行优化流程
    (optimizer
     .load_model()
     .apply_pruning(pruning_amount=0.3)
     .optimize_attention()
     .quantize_model()
     .export_to_onnx()
     .verify_performance())
    
    print("\nModel optimization completed!")

if __name__ == "__main__":
    optimize_model()