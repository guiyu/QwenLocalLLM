# 文件路径: utils/resource_manager.py

import logging
import psutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from .exceptions import ResourceError

logger = logging.getLogger(__name__)

class ResourceManager:
    """资源管理器"""
    def __init__(self, config):
        self.config = config
        self.resources = {}
        self.locks = {}
        self._monitor_thread = None
        self._monitoring = False
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """开始资源监控"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self) -> None:
        """停止资源监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Resource monitoring stopped")
        
    def _monitor_resources(self, interval: float) -> None:
        """资源监控主循环"""
        while self._monitoring:
            try:
                # 获取系统资源使用情况
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 更新资源信息
                self.resources = {
                    'cpu': {
                        'percent': cpu_percent,
                        'count': psutil.cpu_count(),
                        'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
                    },
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent
                    },
                    'disk': {
                        'total': disk.total,
                        'free': disk.free,
                        'percent': disk.percent
                    }
                }
                
                # 检查资源限制
                self._check_resource_limits()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval)
                
    def _check_resource_limits(self) -> None:
        """检查资源限制"""
        try:
            # 检查内存使用
            if self.resources['memory']['percent'] > 90:
                self._handle_memory_pressure()
                
            # 检查CPU使用
            if self.resources['cpu']['percent'] > 80:
                self._handle_cpu_pressure()
                
            # 检查磁盘使用
            if self.resources['disk']['percent'] > 95:
                self._handle_disk_pressure()
                
        except Exception as e:
            logger.error(f"Resource limit check failed: {e}")
            
    def _handle_memory_pressure(self) -> None:
        """处理内存压力"""
        logger.warning("Memory pressure detected")
        # 通知垃圾回收
        import gc
        gc.collect()
        
    def _handle_cpu_pressure(self) -> None:
        """处理CPU压力"""
        logger.warning("CPU pressure detected")
        
    def _handle_disk_pressure(self) -> None:
        """处理磁盘压力"""
        logger.warning("Disk pressure detected")
        
    def allocate_resource(self, resource_type: str, 
                         amount: Union[int, float]) -> bool:
        """分配资源"""
        try:
            if resource_type == 'memory':
                return self._allocate_memory(amount)
            elif resource_type == 'cpu':
                return self._allocate_cpu(amount)
            elif resource_type == 'disk':
                return self._allocate_disk(amount)
            else:
                logger.error(f"Unknown resource type: {resource_type}")
                return False
        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")
            return False
            
    def _allocate_memory(self, amount: int) -> bool:
        """分配内存"""
        try:
            available = psutil.virtual_memory().available
            if amount > available:
                logger.warning(f"Not enough memory: requested {amount}, available {available}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Memory allocation failed: {e}")
            return False
            
    def _allocate_cpu(self, cores: int) -> bool:
        """分配CPU核心"""
        try:
            available_cores = psutil.cpu_count()
            if cores > available_cores:
                logger.warning(f"Not enough CPU cores: requested {cores}, available {available_cores}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"CPU allocation failed: {e}")
            return False
            
    def _allocate_disk(self, size: int) -> bool:
        """分配磁盘空间"""
        try:
            available = psutil.disk_usage('/').free
            if size > available:
                logger.warning(f"Not enough disk space: requested {size}, available {available}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Disk allocation failed: {e}")
            return False
            
    def get_resource_usage(self) -> Dict:
        """获取资源使用情况"""
        return self.resources
        
    def check_resource_availability(self, requirements: Dict) -> bool:
        """检查资源可用性"""
        try:
            for resource_type, amount in requirements.items():
                if not self.allocate_resource(resource_type, amount):
                    return False
            return True
        except Exception as e:
            logger.error(f"Resource availability check failed: {e}")
            return False
            
    def generate_resource_report(self) -> Dict:
            """生成资源报告"""
            try:
                report = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'resources': self.get_resource_usage(),
                    'limits': {
                        'memory': self.config.MEMORY_CONFIG,
                        'storage': self.config.PERFORMANCE_TARGETS['storage_size']
                    },
                    'warnings': [],
                    'recommendations': []
                }
                
                # 检查内存使用
                memory_percent = self.resources['memory']['percent']
                if memory_percent > 80:
                    report['warnings'].append(f"High memory usage: {memory_percent}%")
                    report['recommendations'].append(
                        "Consider implementing more aggressive memory cleanup"
                    )
                
                # 检查CPU使用
                cpu_percent = self.resources['cpu']['percent']
                if cpu_percent > 70:
                    report['warnings'].append(f"High CPU usage: {cpu_percent}%")
                    report['recommendations'].append(
                        "Consider optimizing thread allocation and workload distribution"
                    )
                
                # 检查磁盘使用
                disk_percent = self.resources['disk']['percent']
                if disk_percent > 85:
                    report['warnings'].append(f"High disk usage: {disk_percent}%")
                    report['recommendations'].append(
                        "Consider cleaning up unused model files and caches"
                    )
                
                return report
                
            except Exception as e:
                logger.error(f"Failed to generate resource report: {e}")
                return {}
            
    def save_resource_report(self, report: Dict, 
                           filepath: Optional[str] = None) -> bool:
        """保存资源报告"""
        try:
            if filepath is None:
                report_dir = self.config.PROJECT_ROOT / "reports"
                report_dir.mkdir(exist_ok=True)
                filepath = report_dir / f"resource_report_{int(time.time())}.json"
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Resource report saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save resource report: {e}")
            return False
            
    def get_cpu_info(self) -> Dict:
        """获取CPU详细信息"""
        try:
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'frequency': {}
            }
            
            # CPU频率信息
            freq = psutil.cpu_freq()
            if freq:
                cpu_info['frequency'] = {
                    'current': freq.current,
                    'min': freq.min,
                    'max': freq.max
                }
            
            # CPU使用率
            cpu_info['usage'] = {
                'overall': psutil.cpu_percent(interval=1),
                'per_core': psutil.cpu_percent(interval=1, percpu=True)
            }
            
            return cpu_info
            
        except Exception as e:
            logger.error(f"Failed to get CPU info: {e}")
            return {}
            
    def get_memory_info(self) -> Dict:
        """获取内存详细信息"""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'virtual': {
                    'total': mem.total,
                    'available': mem.available,
                    'used': mem.used,
                    'free': mem.free,
                    'percent': mem.percent,
                    'active': getattr(mem, 'active', None),
                    'inactive': getattr(mem, 'inactive', None),
                    'buffers': getattr(mem, 'buffers', None),
                    'cached': getattr(mem, 'cached', None)
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}
            
    def optimize_resource_usage(self) -> bool:
        """优化资源使用"""
        try:
            optimizations_applied = []
            
            # 1. 检查并优化内存使用
            if self.resources['memory']['percent'] > 75:
                success = self._optimize_memory_usage()
                if success:
                    optimizations_applied.append('memory')
            
            # 2. 检查并优化CPU使用
            if self.resources['cpu']['percent'] > 70:
                success = self._optimize_cpu_usage()
                if success:
                    optimizations_applied.append('cpu')
            
            # 3. 检查并优化磁盘使用
            if self.resources['disk']['percent'] > 80:
                success = self._optimize_disk_usage()
                if success:
                    optimizations_applied.append('disk')
            
            if optimizations_applied:
                logger.info(f"Applied optimizations for: {', '.join(optimizations_applied)}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return False
            
    def _optimize_memory_usage(self) -> bool:
        """优化内存使用"""
        try:
            # 1. 强制垃圾回收
            import gc
            gc.collect()
            
            # 2. 清理Python对象缓存
            import sys
            sys.exc_clear()
            
            # 3. 释放未使用的内存给操作系统
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
            
            return True
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return False
            
    def _optimize_cpu_usage(self) -> bool:
        """优化CPU使用"""
        try:
            # 实现CPU使用优化策略
            return True
            
        except Exception as e:
            logger.error(f"CPU optimization failed: {e}")
            return False
            
    def _optimize_disk_usage(self) -> bool:
        """优化磁盘使用"""
        try:
            # 实现磁盘使用优化策略
            return True
            
        except Exception as e:
            logger.error(f"Disk optimization failed: {e}")
            return False