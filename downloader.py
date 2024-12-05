import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def download_qwen_model():
    # 设置模型缓存目录
    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # 使用Qwen-2.5-0.5B版本
    model_name = "Qwen/Qwen2.5-0.5B"

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16  # 使用FP16以减少内存占用
    )

    # 保存模型和分词器到本地
    output_dir = "./qwen_small_model"
    print(f"Saving model and tokenizer to {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = download_qwen_model()
    print("Model download completed!")

    # 打印模型大小信息
    model_size = sum(p.numel() for p in model.parameters()) * 2 / (1024 * 1024 * 1024)  # in GB
    print(f"Model size in FP16: {model_size:.2f} GB")