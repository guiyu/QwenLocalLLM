# 文件路径: utils/helpers.py

import os
from pathlib import Path
import hashlib
import logging
import shutil
import torch

logger = logging.getLogger(__name__)

class Utils:
    @staticmethod
    def ensure_directory(directory: Path) -> bool:
        """确保目录存在"""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False

    @staticmethod
    def calculate_md5(file_path: Path) -> str:
        """计算文件MD5"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate MD5 for {file_path}: {e}")
            return ""

    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """获取文件大小（字节）"""
        try:
            return file_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            return 0

    @staticmethod
    def get_available_device() -> torch.device:
        """获取可用的计算设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def copy_file(src: Path, dst: Path) -> bool:
        """复制文件"""
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            return True
        except Exception as e:
            logger.error(f"Failed to copy file from {src} to {dst}: {e}")
            return False

    @staticmethod
    def remove_file(path: Path) -> bool:
        """删除文件"""
        try:
            if path.exists():
                path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to remove file {path}: {e}")
            return False

    @staticmethod
    def get_memory_info() -> dict:
        """获取内存使用情况"""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                'total': vm.total,
                'available': vm.available,
                'percent': vm.percent,
                'used': vm.used,
                'free': vm.free
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
    
    def verify_model_files(self, model_dir: Path) -> bool:
        """验证模型文件的完整性和正确性"""
        try:
            required_files = {
                "model_quantized.onnx": {
                    "min_size": 100 * 1024 * 1024,  # 100MB最小大小
                    "signature": b"ONNX"  # ONNX文件标识
                },
                "vocab.json": {
                    "min_size": 100 * 1024,  # 100KB最小大小
                    "required_keys": ["model_type", "vocab"]
                },
                "config.json": {
                    "min_size": 1024,  # 1KB最小大小
                    "required_keys": ["model_type", "architectures"]
                }
            }

            for filename, checks in required_files.items():
                file_path = model_dir / filename
                if not file_path.exists():
                    logger.error(f"Missing required file: {filename}")
                    return False

                # 检查文件大小
                if file_path.stat().st_size < checks["min_size"]:
                    logger.error(f"File too small: {filename}")
                    return False

                # 检查文件格式
                with open(file_path, "rb") as f:
                    if filename.endswith(".onnx"):
                        # 验证ONNX文件头
                        header = f.read(4)
                        if header != checks["signature"]:
                            logger.error(f"Invalid ONNX file: {filename}")
                            return False
                    elif filename.endswith(".json"):
                        # 验证JSON文件结构
                        import json
                        try:
                            data = json.load(f)
                            for key in checks["required_keys"]:
                                if key not in data:
                                    logger.error(f"Missing key in {filename}: {key}")
                                    return False
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON file: {filename}")
                            return False

            return True
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False