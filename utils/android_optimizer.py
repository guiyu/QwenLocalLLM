# 文件路径: utils/android_optimizer.py

import logging
import psutil
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AndroidOptimizer:
    """Android优化工具类"""
    def __init__(self, config):
        self.config = config
        self.mtk_config = config.MTK_CONFIG
        
    def optimize_memory_alignment(self, data: bytes) -> bytes:
        """内存对齐优化"""
        # MTK要求16字节对齐
        alignment = self.mtk_config['memory_alignment']
        if len(data) % alignment != 0:
            padding = alignment - (len(data) % alignment)
            data += b'\0' * padding
        return data
        
    def optimize_thread_allocation(self) -> Dict[str, List[int]]:
        """线程分配优化"""
        try:
            # 获取CPU信息
            cpu_count = psutil.cpu_count()
            if cpu_count <= 2:
                # 单核或双核设备
                return {
                    'inference': [0],
                    'preprocessing': [0],
                    'postprocessing': [0]
                }
            elif cpu_count <= 4:
                # 四核设备
                return {
                    'inference': [2, 3],  # 使用大核心
                    'preprocessing': [0],  # 使用小核心
                    'postprocessing': [1]  # 使用小核心
                }
            else:
                # 多核设备
                return {
                    'inference': [4, 5, 6, 7],  # 使用大核心
                    'preprocessing': [0, 1],    # 使用小核心
                    'postprocessing': [2, 3]    # 使用小核心
                }
        except Exception as e:
            logger.error(f"Failed to optimize thread allocation: {e}")
            # 返回默认配置
            return {
                'inference': [0],
                'preprocessing': [0],
                'postprocessing': [0]
            }
            
    def optimize_memory_usage(self) -> Dict:
        """内存使用优化"""
        try:
            total_memory = psutil.virtual_memory().total
            available_memory = psutil.virtual_memory().available
            
            # 计算每个组件的内存限制
            memory_limits = {
                'model': min(
                    self.config.MEMORY_CONFIG['model_cache'],
                    available_memory * 0.6  # 最多使用60%可用内存
                ),
                'kv_cache': min(
                    self.config.MEMORY_CONFIG['kv_cache'],
                    available_memory * 0.2  # 最多使用20%可用内存
                ),
                'working': min(
                    available_memory * 0.1,  # 最多使用10%可用内存
                    512 * 1024 * 1024  # 512MB上限
                )
            }
            
            return {
                'limits': memory_limits,
                'total_memory': total_memory,
                'available_memory': available_memory
            }
        except Exception as e:
            logger.error(f"Failed to optimize memory usage: {e}")
            return self.config.MEMORY_CONFIG
            
    def optimize_model_loading(self, model_path: Path) -> Dict:
        """模型加载优化"""
        try:
            file_size = model_path.stat().st_size
            
            # 计算加载参数
            chunk_size = self.mtk_config['chunk_size']
            num_chunks = (file_size + chunk_size - 1) // chunk_size
            
            return {
                'chunk_size': chunk_size,
                'num_chunks': num_chunks,
                'file_size': file_size,
                'alignment': self.mtk_config['memory_alignment']
            }
        except Exception as e:
            logger.error(f"Failed to optimize model loading: {e}")
            return {
                'chunk_size': 1024 * 1024,  # 1MB默认值
                'num_chunks': 1,
                'file_size': 0,
                'alignment': 16
            }
            
    def get_device_capabilities(self) -> Dict:
        """获取设备能力"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_usage': psutil.disk_usage('/').free
            }
        except Exception as e:
            logger.error(f"Failed to get device capabilities: {e}")
            return {}
            
    def generate_optimization_config(self) -> Dict:
        """生成优化配置"""
        device_caps = self.get_device_capabilities()
        memory_config = self.optimize_memory_usage()
        thread_config = self.optimize_thread_allocation()
        
        return {
            'device': device_caps,
            'memory': memory_config,
            'threading': thread_config,
            'mtk_specific': self.mtk_config,
            'model_loading': {
                'chunk_size': self.mtk_config['chunk_size'],
                'page_size': self.mtk_config['page_size'],
                'alignment': self.mtk_config['memory_alignment']
            }
        }