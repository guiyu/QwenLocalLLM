# 文件路径: utils/helpers.py

import os
import torch
import hashlib
import json
from pathlib import Path
import shutil
from typing import Optional, Union, Dict, Any
import psutil
import logging

logger = logging.getLogger(__name__)

class Utils:
    @staticmethod
    def get_available_device() -> torch.device:
        """获取可用的计算设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def calculate_md5(file_path: Union[str, Path]) -> str:
        """计算文件MD5"""
        hash_md5 = hashlib.md5()
        with open(str(file_path), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """保存JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}")
            return False

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            return None

    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """获取内存使用信息"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  # MB
            'vms': memory_info.vms / (1024 * 1024),  # MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / (1024 * 1024)  # MB
        }

    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> bool:
        """确保目录存在"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return False

    @staticmethod
    def clean_directory(directory: Union[str, Path], 
                       exclude: Optional[list] = None) -> bool:
        """清理目录"""
        try:
            directory = Path(directory)
            if not directory.exists():
                return True

            exclude = exclude or []
            for item in directory.iterdir():
                if item.name not in exclude:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            return True
        except Exception as e:
            logger.error(f"Failed to clean directory: {e}")
            return False

    @staticmethod
    def copy_with_progress(src: Union[str, Path], dst: Union[str, Path], 
                          chunk_size: int = 1024*1024) -> bool:
        """带进度的文件复制"""
        try:
            src, dst = Path(src), Path(dst)
            
            # 确保目标目录存在
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # 获取文件大小
            file_size = src.stat().st_size
            
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                copied = 0
                while True:
                    buf = fsrc.read(chunk_size)
                    if not buf:
                        break
                    fdst.write(buf)
                    copied += len(buf)
                    progress = (copied / file_size) * 100
                    print(f"\rCopying: {progress:.1f}%", end='')
                    
            print()  # 换行
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False

    @staticmethod
    def get_file_size(file_path: Union[str, Path], unit: str = 'MB') -> float:
        """获取文件大小"""
        try:
            size_bytes = Path(file_path).stat().st_size
            units = {
                'B': 1,
                'KB': 1024,
                'MB': 1024*1024,
                'GB': 1024*1024*1024
            }
            return size_bytes / units.get(unit.upper(), 1)
        except Exception as e:
            logger.error(f"Failed to get file size: {e}")
            return 0.0

    @staticmethod
    def is_process_running(process_name: str) -> bool:
        """检查进程是否运行"""
        for proc in psutil.process_iter(['name']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, 
                    psutil.ZombieProcess):
                pass
        return False

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """获取系统信息"""
        import platform
        
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'total_memory': psutil.virtual_memory().total / (1024*1024*1024),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else None
        }

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"