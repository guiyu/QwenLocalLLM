# 文件路径: script/optimize/pruning_optimizer.py
# 更新文件，添加更多剪枝策略

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ModelPruner:
    def __init__(self, model_path, output_dir="./models/pruned"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        """加载模型"""
        logger.info(f"Loading model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        return self
    
    def analyze_model(self):
        """分析模型参数分布"""
        total_params = 0
        layer_params = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                params = module.weight.numel()
                total_params += params
                layer_params[name] = params
        
        return total_params, layer_params
    
    def structured_pruning(self, pruning_amount=0.3):
        """结构化剪枝"""
        logger.info(f"Applying structured pruning with amount {pruning_amount}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=pruning_amount
                )
                prune.remove(module, 'weight')
        
        return self
    
    def gradual_pruning(self, initial_amount=0.1, final_amount=0.3, steps=3):
        """渐进式剪枝"""
        logger.info("Starting gradual pruning")
        
        step_amount = (final_amount - initial_amount) / steps
        current_amount = initial_amount
        
        for step in range(steps):
            logger.info(f"Pruning step {step+1}/{steps}, amount={current_amount:.3f}")
            self.structured_pruning(current_amount)
            current_amount += step_amount
        
        return self
    
    def save_pruned_model(self):
        """保存剪枝后的模型"""
        output_path = self.output_dir / "pruned_model"
        self.model.save_pretrained(str(output_path))
        
        # 保存剪枝信息
        total_params, layer_params = self.analyze_model()
        pruning_info = {
            "total_parameters": total_params,
            "layer_parameters": layer_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # 假设FP32
        }
        
        with open(self.output_dir / "pruning_info.json", "w") as f:
            json.dump(pruning_info, f, indent=2)
        
        logger.info(f"Saved pruned model to {output_path}")
        return str(output_path)

def prune_model(model_path, output_dir, config):
    import onnx
    from onnx import helper
    from pathlib import Path

    # 加载 ONNX 模型
    model = onnx.load(model_path)
    print(f"Loaded ONNX model from: {model_path}")

    # 定义需要保留的节点条件
    retained_node_names = set()
    retained_tensors = set()

    # 选择保留的节点（例如 attention 和 layer_norm）
    for node in model.graph.node:
        if "attention" in node.name or "layer_norm" in node.name:
            retained_node_names.add(node.name)

    # 递归追踪依赖
    def collect_dependencies(tensor_name):
        for node in model.graph.node:
            if tensor_name in node.output:
                retained_node_names.add(node.name)
                for input_name in node.input:
                    retained_tensors.add(input_name)
                    collect_dependencies(input_name)

    # 为每个节点收集依赖
    for node_name in list(retained_node_names):
        for node in model.graph.node:
            if node.name == node_name:
                for input_name in node.input:
                    retained_tensors.add(input_name)
                    collect_dependencies(input_name)

    # 保留节点和初始化张量
    pruned_nodes = [node for node in model.graph.node if node.name in retained_node_names]
    pruned_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in retained_tensors
    ]
    pruned_value_infos = [
        value_info for value_info in model.graph.value_info if value_info.name in retained_tensors
    ]

    # 校正拓扑排序
    sorted_nodes = []
    visited = set()

    def topological_sort(node_name):
        if node_name in visited:
            return
        visited.add(node_name)
        for node in pruned_nodes:
            if node.name == node_name:
                for input_name in node.input:
                    if input_name in retained_tensors:
                        topological_sort(input_name)  # 递归排序依赖
                sorted_nodes.append(node)

    for node in pruned_nodes:
        topological_sort(node.name)

    # 构建新的模型图
    pruned_graph = helper.make_graph(
        nodes=sorted_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=pruned_initializers,
        value_info=pruned_value_infos,
    )
    pruned_model = helper.make_model(pruned_graph)

    # 保存剪枝后的模型
    output_path = Path(output_dir) / "pruned_model.onnx"
    onnx.save(pruned_model, str(output_path))
    print(f"Pruned model saved to: {output_path}")
    return output_path