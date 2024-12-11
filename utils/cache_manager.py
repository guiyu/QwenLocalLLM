# 文件路径: utils/cache_manager.py

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import OrderedDict
import threading
import json
import shutil

logger = logging.getLogger(__name__)

class LRUCache:
    """LRU缓存实现"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
            
    def put(self, key: str, value: Any) -> None:
        """添加缓存项"""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
                
    def remove(self, key: str) -> None:
        """移除缓存项"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()

class CacheManager:
    """缓存管理器"""
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(self.config.HF_CACHE_DIR)
        self.model_cache = LRUCache(5)  # 最多缓存5个模型
        self.data_cache = {}  # 数据缓存
        self.cache_info = {}  # 缓存信息
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载缓存信息
        self._load_cache_info()
        
    def _load_cache_info(self) -> None:
        """加载缓存信息"""
        cache_info_file = self.cache_dir / "cache_info.json"
        try:
            if cache_info_file.exists():
                with open(cache_info_file, 'r') as f:
                    self.cache_info = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache info: {e}")
            self.cache_info = {}
            
    def _save_cache_info(self) -> None:
        """保存缓存信息"""
        try:
            cache_info_file = self.cache_dir / "cache_info.json"
            with open(cache_info_file, 'w') as f:
                json.dump(self.cache_info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache info: {e}")
            
    def add_to_cache(self, key: str, data: Any, 
                     cache_type: str = 'data') -> bool:
        """添加到缓存"""
        try:
            if cache_type == 'model':
                self.model_cache.put(key, data)
            else:
                self.data_cache[key] = data
                
            # 更新缓存信息
            self.cache_info[key] = {
                'type': cache_type,
                'time': time.time(),
                'size': self._get_size(data)
            }
            self._save_cache_info()
            return True
        except Exception as e:
            logger.error(f"Failed to add to cache: {e}")
            return False
            
    def get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        try:
            if key not in self.cache_info:
                return None
                
            cache_type = self.cache_info[key]['type']
            if cache_type == 'model':
                return self.model_cache.get(key)
            else:
                return self.data_cache.get(key)
        except Exception as e:
            logger.error(f"Failed to get from cache: {e}")
            return None
            
    def remove_from_cache(self, key: str) -> bool:
        """从缓存移除数据"""
        try:
            if key not in self.cache_info:
                return False
                
            cache_type = self.cache_info[key]['type']
            if cache_type == 'model':
                self.model_cache.remove(key)
            else:
                self.data_cache.pop(key, None)
                
            del self.cache_info[key]
            self._save_cache_info()
            return True
        except Exception as e:
            logger.error(f"Failed to remove from cache: {e}")
            return False
            
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """清空缓存"""
        try:
            if cache_type == 'model':
                self.model_cache.clear()
            elif cache_type == 'data':
                self.data_cache.clear()
            else:
                self.model_cache.clear()
                self.data_cache.clear()
                
            # 清理缓存信息
            if cache_type:
                self.cache_info = {k: v for k, v in self.cache_info.items()
                                 if v['type'] != cache_type}
            else:
                self.cache_info.clear()
                
            self._save_cache_info()
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
            
    def _get_size(self, obj: Any) -> int:
        """获取对象大小"""
        if isinstance(obj, (str, bytes)):
            return len(obj)
        elif isinstance(obj, Path):
            return obj.stat().st_size
        else:
            return 0
            
    def cleanup_old_cache(self, max_age: int = 7*24*60*60) -> None:
        """清理旧缓存"""
        try:
            current_time = time.time()
            keys_to_remove = []
            
            for key, info in self.cache_info.items():
                if current_time - info['time'] > max_age:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                self.remove_from_cache(key)
                
            logger.info(f"Removed {len(keys_to_remove)} old cache entries")
        except Exception as e:
            logger.error(f"Failed to cleanup old cache: {e}")
            
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        try:
            stats = {
                'model_cache_size': len(self.model_cache.cache),
                'data_cache_size': len(self.data_cache),
                'total_entries': len(self.cache_info),
                'cache_types': {},
                'total_size': 0
            }
            
            # 统计各类型缓存
            for info in self.cache_info.values():
                cache_type = info['type']
                stats['cache_types'][cache_type] = stats['cache_types'].get(cache_type, 0) + 1
                stats['total_size'] += info.get('size', 0)
                
            return stats
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
            
    def export_cache_info(self, filepath: Optional[str] = None) -> bool:
        """导出缓存信息"""
        try:
            if filepath is None:
                filepath = self.cache_dir / f"cache_export_{int(time.time())}.json"
                
            cache_export = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stats': self.get_cache_stats(),
                'entries': self.cache_info
            }
            
            with open(filepath, 'w') as f:
                json.dump(cache_export, f, indent=2)
                
            logger.info(f"Cache info exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export cache info: {e}")
            return False