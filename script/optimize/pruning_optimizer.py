import torch
from transformers import AutoModelForCausalLM
import torch.nn.utils.prune as prune


def apply_pruning(model, pruning_amount=0.3):
    """
    对模型进行结构化剪枝
    pruning_amount: 要剪掉的权重比例
    """
    for name, module in model.named_modules():
        # 对线性层和注意力层进行剪枝
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(
                module,
                name='weight',
                amount=pruning_amount
            )
            # 使剪枝永久化
            prune.remove(module, 'weight')

    return model


def prune_model():
    # 加载微调后的模型
    model_path = "./fine_tuned_model"
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print("Starting model pruning...")

    # 应用剪枝
    pruned_model = apply_pruning(model)

    # 评估剪枝后的模型大小
    original_size = sum(p.numel() for p in model.parameters())
    pruned_size = sum(p.numel() for p in pruned_model.parameters())

    print(f"Original model size: {original_size:,} parameters")
    print(f"Pruned model size: {pruned_size:,} parameters")
    print(f"Size reduction: {(1 - pruned_size / original_size) * 100:.2f}%")

    # 保存剪枝后的模型
    output_dir = "./pruned_model"
    pruned_model.save_pretrained(output_dir)
    print(f"Pruned model saved to {output_dir}")

    return pruned_model


if __name__ == "__main__":
    pruned_model = prune_model()