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
    try:
        logger.info("Starting model pruning process...")
        
        pruner = ModelPruner(model_path, output_dir)
        pruned_model_path = (
            pruner.load_model()
            .gradual_pruning(
                initial_amount=config.INITIAL_PRUNING_AMOUNT,
                final_amount=config.FINAL_PRUNING_AMOUNT,
                steps=config.PRUNING_STEPS
            )
            .save_pruned_model()
        )
        
        logger.info("Model pruning completed successfully!")
        return pruned_model_path
        
    except Exception as e:
        logger.error(f"Error during pruning: {str(e)}")
        raise