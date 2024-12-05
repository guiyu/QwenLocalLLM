import torch
import torch.quantization
from transformers import AutoModelForCausalLM
import onnx
import onnxruntime as ort


def quantize_and_export_model():
    # 加载剪枝后的模型
    model_path = "./pruned_model"
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print("Starting model quantization...")

    # 将模型设置为评估模式
    model.eval()

    # 准备示例输入
    dummy_input = torch.randint(100, (1, 128))  # 批次大小为1，序列长度为128

    # 导出为ONNX格式
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )

    # 量化ONNX模型
    print("Quantizing ONNX model...")
    from onnxruntime.quantization import quantize_dynamic

    quantize_dynamic(
        "model.onnx",
        "model_quantized.onnx",
        weight_type=torch.qint8
    )

    # 验证量化模型的大小
    import os
    original_size = os.path.getsize("model.onnx") / (1024 * 1024)  # MB
    quantized_size = os.path.getsize("model_quantized.onnx") / (1024 * 1024)  # MB

    print(f"Original ONNX model size: {original_size:.2f} MB")
    print(f"Quantized ONNX model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")

    print("Model quantization completed!")


if __name__ == "__main__":
    quantize_and_export_model()