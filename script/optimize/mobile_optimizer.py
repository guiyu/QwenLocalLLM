import torch
import torch.quantization
from transformers import AutoModelForCausalLM
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def optimize_for_mobile():
    # 加载剪枝后的模型
    model_path = "./pruned_model"
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print("Starting mobile optimization...")

    # 1. 导出为ONNX格式，启用优化
    dummy_input = torch.randint(100, (1, 64))  # 减小序列长度以适应移动端

    # 设置动态轴以支持可变长度输入
    dynamic_axes = {
        'input': {0: 'batch', 1: 'sequence'},
        'output': {0: 'batch', 1: 'sequence'}
    }

    # 导出时启用优化
    torch.onnx.export(
        model,
        dummy_input,
        "model_mobile.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,  # 常量折叠优化
        opset_version=13,
        optimize_for_mobile=True
    )

    # 2. 应用量化，使用更激进的设置
    quantize_dynamic(
        "model_mobile.onnx",
        "model_mobile_quantized.onnx",
        weight_type=QuantType.QInt8,  # 使用Int8量化
        optimize_for_mobile=True,
        per_channel=False,  # 使用per-tensor量化以减小模型大小
        reduce_range=True,  # 牺牲一定精度换取更小的模型体积
    )

    # 3. 验证和优化模型大小
    import os
    original_size = os.path.getsize("model_mobile.onnx") / (1024 * 1024)
    quantized_size = os.path.getsize("model_mobile_quantized.onnx") / (1024 * 1024)

    print(f"Original mobile model size: {original_size:.2f} MB")
    print(f"Quantized mobile model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")

    # 4. 验证推理性能
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimize_for_mobile = True

    session = ort.InferenceSession(
        "model_mobile_quantized.onnx",
        session_options,
        providers=['CPUExecutionProvider']
    )

    # 测试推理延迟
    import time
    input_data = {'input': dummy_input.numpy()}

    # 预热运行
    session.run(None, input_data)

    # 测试延迟
    times = []
    for _ in range(10):
        start = time.time()
        session.run(None, input_data)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time * 1000:.2f} ms")


if __name__ == "__main__":
    optimize_for_mobile()