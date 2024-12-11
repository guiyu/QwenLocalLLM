# 文件路径: utils/performance_monitor.py

import time
import logging
import psutil
import threading
from typing import Dict, List, Optional, Callable
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMetric:
    """性能指标类"""
    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.values = deque(maxlen=window_size)
        self.min_value = float('inf')
        self.max_value = float('-inf')
        self.total = 0
        self.count = 0
        
    def add(self, value: float) -> None:
        """添加测量值"""
        self.values.append(value)
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.total += value
        self.count += 1
        
    def get_stats(self) -> Dict:
        """获取统计信息"""
        values = list(self.values)
        return {
            'current': values[-1] if values else 0,
            'min': self.min_value if self.count else 0,
            'max': self.max_value if self.count else 0,
            'avg': self.total / self.count if self.count else 0,
            'p90': np.percentile(values, 90) if values else 0,
            'p95': np.percentile(values, 95) if values else 0,
            'p99': np.percentile(values, 99) if values else 0
        }

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, config):
        self.config = config
        self.metrics = {}
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """开始监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
        logger.info("Performance monitoring stopped")
        
    def add_callback(self, callback: Callable) -> None:
        """添加回调函数"""
        self.callbacks.append(callback)
        
    def get_metrics(self) -> Dict:
        """获取所有指标"""
        return {name: metric.get_stats() 
                for name, metric in self.metrics.items()}
        
    def record_metric(self, name: str, value: float) -> None:
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetric(name)
        self.metrics[name].add(value)
        
    def _monitor_loop(self, interval: float) -> None:
        """监控循环"""
        while self.monitoring:
            try:
                # 系统指标
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # 记录指标
                self.record_metric('cpu_usage', cpu_percent)
                self.record_metric('memory_usage', memory.percent)
                
                # 调用回调函数
                metrics = self.get_metrics()
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(interval)
                
    def benchmark_inference(self, model_path: str, 
                          num_runs: int = 100) -> Dict:
        """推理性能基准测试"""
        try:
            import onnxruntime as ort
            
            # 创建推理会话
            session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # 准备输入数据
            input_name = session.get_inputs()[0].name
            input_shape = (1, 32)
            input_data = np.random.randint(0, 100, input_shape).astype(np.int64)
            
            # 预热
            for _ in range(10):
                _ = session.run(None, {input_name: input_data})
            
            # 测试运行
            latencies = []
            memory_usage = []
            
            for _ in range(num_runs):
                # 记录内存使用
                memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_usage.append(memory)
                
                # 记录推理时间
                start_time = time.time()
                _ = session.run(None, {input_name: input_data})
                latencies.append((time.time() - start_time) * 1000)  # ms
            
            # 计算统计信息
            results = {
                'latency': {
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies),
                    'p50': np.percentile(latencies, 50),
                    'p90': np.percentile(latencies, 90),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99)
                },
                'memory': {
                    'mean': np.mean(memory_usage),
                    'max': np.max(memory_usage)
                }
            }
            
            logger.info("Benchmark results:")
            logger.info(f"Mean latency: {results['latency']['mean']:.2f}ms")
            logger.info(f"P90 latency: {results['latency']['p90']:.2f}ms")
            logger.info(f"Mean memory: {results['memory']['mean']:.2f}MB")
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {}
            
    def export_metrics(self, filepath: Optional[str] = None) -> None:
        """导出指标"""
        if filepath is None:
            filepath = f"metrics_{int(time.time())}.json"
            
        try:
            metrics = self.get_metrics()
            metrics = self.get_metrics()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            
    def verify_performance(self) -> Dict[str, bool]:
        """验证性能是否满足目标"""
        results = {}
        metrics = self.get_metrics()
        
        # 验证推理延迟
        if 'inference_latency' in metrics:
            latency_stats = metrics['inference_latency']
            results['latency'] = (
                latency_stats['p90'] <= self.config.PERFORMANCE_TARGETS['inference_time']
            )
            
        # 验证内存使用
        if 'memory_usage' in metrics:
            memory_stats = metrics['memory_usage']
            results['memory'] = (
                memory_stats['max'] <= self.config.PERFORMANCE_TARGETS['memory_usage']
            )
            
        # 验证模型大小
        model_path = self.config.get_model_paths()['quantized_model']
        if model_path.exists():
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            results['model_size'] = (
                model_size <= self.config.PERFORMANCE_TARGETS['storage_size']
            )
            
        return results
        
    def generate_report(self, include_recommendations: bool = True) -> Dict:
        """生成性能报告"""
        metrics = self.get_metrics()
        verification = self.verify_performance()
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'verification': verification,
            'summary': {
                'passed': all(verification.values()),
                'failed_checks': [
                    k for k, v in verification.items() if not v
                ]
            }
        }
        
        if include_recommendations:
            report['recommendations'] = self._generate_recommendations(
                metrics, verification
            )
            
        return report
        
    def _generate_recommendations(self, metrics: Dict, 
                                verification: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 检查延迟问题
        if not verification.get('latency', True):
            latency_stats = metrics.get('inference_latency', {})
            if latency_stats.get('p90', 0) > self.config.PERFORMANCE_TARGETS['inference_time']:
                recommendations.append(
                    "考虑进一步量化或优化模型来减少推理延迟"
                )
                
        # 检查内存问题
        if not verification.get('memory', True):
            memory_stats = metrics.get('memory_usage', {})
            if memory_stats.get('max', 0) > self.config.PERFORMANCE_TARGETS['memory_usage']:
                recommendations.append(
                    "考虑优化内存管理策略，可能需要增加内存清理频率"
                )
                
        # 检查模型大小问题
        if not verification.get('model_size', True):
            recommendations.append(
                "考虑使用更激进的量化策略或模型剪枝来减小模型大小"
            )
            
        return recommendations
        
    def log_performance_metrics(self):
        """记录性能指标到日志"""
        metrics = self.get_metrics()
        verification = self.verify_performance()
        logger.info("Performance Metrics Summary:")
        
        for name, stats in metrics.items():
            logger.info(f"\n{name}:")
            for stat_name, value in stats.items():
                logger.info(f"  {stat_name}: {value:.2f}")
                
        logger.info("\nPerformance Verification:")
        for check, passed in verification.items():
            status = "PASSED" if passed else "FAILED"
            logger.info(f"{check}: {status}")
            
        if not all(verification.values()):
            recommendations = self._generate_recommendations(
                metrics, verification
            )
            logger.info("\nRecommendations:")
            for rec in recommendations:
                logger.info(f"- {rec}")