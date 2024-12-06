# 文件路径: script/download/model_downloader.py
# 修改说明: 添加进度条和错误处理

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_qwen_model(model_version="Qwen/Qwen2.5-0.5B", output_dir="./models/original"):
    try:
        # 设置模型缓存目录
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"开始下载模型 {model_version}")
        logger.info("下载tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_version,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        logger.info("下载model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_version,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # 保存模型和tokenizer
        logger.info(f"保存模型到 {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # 打印模型信息
        model_size = sum(p.numel() for p in model.parameters()) * 2 / (1024 * 1024 * 1024)  # GB
        logger.info(f"模型大小: {model_size:.2f} GB")
        logger.info(f"参数数量: {model.num_parameters():,}")

        return model, tokenizer

    except Exception as e:
        logger.error(f"下载模型时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model, tokenizer = download_qwen_model()
        logger.info("模型下载完成！")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")