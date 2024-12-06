# 文件路径: utils/helpers.py
# 新建文件

import os
import hashlib
import requests
from tqdm import tqdm
import torch

class Utils:
    @staticmethod
    def calculate_md5(file_path):
        """计算文件MD5"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def download_file(url, dest_path, expected_md5=None):
        """下载文件并验证MD5"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        if expected_md5:
            actual_md5 = Utils.calculate_md5(dest_path)
            if actual_md5 != expected_md5:
                raise ValueError("MD5 verification failed")
    
    @staticmethod
    def get_available_device():
        """获取可用的计算设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def ensure_directory(directory):
        """确保目录存在"""
        os.makedirs(directory, exist_ok=True)