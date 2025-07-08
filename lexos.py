"""
LexOS Production Enhancements - Section 1: Advanced Architecture & Performance
Production-ready improvements for your AI operating system
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
import numpy as np
from PIL import Image
import io
import base64
import psutil
import GPUtil
from ollama import AsyncClient as OllamaClient
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import yaml
from collections import deque, defaultdict
import heapq
import traceback
from functools import lru_cache, wraps
import time
import networkx as nx
import random
import gc
import os

# Enhanced logging with performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lexos.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LexOS")

# ==================== PERFORMANCE MONITORING ====================

class PerformanceMonitor:
    """Advanced performance monitoring and optimization"""

    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'min_time': float('inf'),
            'max_time': 0,
            'errors': 0,
            'last_10': deque(maxlen=10)
        })
        self.resource_history = deque(maxlen=1000)
        self.bottlenecks = []
        self.last_report = None

    def track(self, operation: str):
        """Decorator to track operation performance"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                error_occurred = False
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_occurred = True
                    self.metrics[operation]['errors'] += 1
                    logger.error(f"Error in {operation}: {e}\n{traceback.format_exc()}")
                    raise
                finally:
                    elapsed = time.time() - start_time
                    metric = self.metrics[operation]
                    metric['count'] += 1
                    metric['total_time'] += elapsed
                    metric['min_time'] = min(metric['min_time'], elapsed)
                    metric['max_time'] = max(metric['max_time'], elapsed)
                    metric['last_10'].append(elapsed)
                    # Detect bottlenecks
                    if elapsed > 5.0 and not error_occurred:
                        self.bottlenecks.append({
                            'operation': operation,
                            'time': elapsed,
                            'timestamp': datetime.now()
                        })
            return wrapper
        return decorator

    async def collect_resource_metrics(self):
        """Collect system resource metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
            'network': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
        }
        # GPU metrics (now support all GPUs, not just first)
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics['gpus'] = [
                {
                    'id': g.id,
                    'name': g.name,
                    'load': g.load,
                    'memory_used': g.memoryUsed,
                    'memory_total': g.memoryTotal,
                    'temperature': g.temperature
                } for g in gpus
            ]
        self.resource_history.append(metrics)
        return metrics

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'operations': {},
            'resource_usage': self._analyze_resource_trends(),
            'bottlenecks': self.bottlenecks[-10:],
            'recommendations': self._generate_recommendations()
        }
        for op, metrics in self.metrics.items():
            avg_time = metrics['total_time'] / metrics['count'] if metrics['count'] > 0 else 0
            recent_avg = sum(metrics['last_10']) / len(metrics['last_10']) if metrics['last_10'] else 0
            report['operations'][op] = {
                'count': metrics['count'],
                'avg_time': avg_time,
                'recent_avg_time': recent_avg,
                'min_time': metrics['min_time'] if metrics['min_time'] != float('inf') else 0,
                'max_time': metrics['max_time'],
                'error_rate': metrics['errors'] / metrics['count'] if metrics['count'] > 0 else 0,
                'trend': 'improving' if recent_avg < avg_time else 'degrading'
            }
        self.last_report = report
        return report

    def _analyze_resource_trends(self) -> Dict[str, Any]:
        """Analyze resource usage trends"""
        if not self.resource_history:
            return {}
        recent = list(self.resource_history)[-100:]
        gpus = [gpu for r in recent for gpu in r.get('gpus', [])]
        return {
            'cpu_avg': np.mean([r['cpu_percent'] for r in recent]),
            'memory_avg': np.mean([r['memory']['percent'] for r in recent]),
            'gpu_memory_avg': np.mean([g['memory_used'] for g in gpus]) if gpus else 0,
            'peak_cpu': max(r['cpu_percent'] for r in recent),
            'peak_memory': max(r['memory']['percent'] for r in recent)
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        if self.bottlenecks:
            slow_ops = defaultdict(int)
            for b in self.bottlenecks:
                slow_ops[b['operation']] += 1
            most_common = max(slow_ops.items(), key=lambda x: x[1])
            recommendations.append(f"Optimize '{most_common[0]}' - caused {most_common[1]} slowdowns")
        trends = self._analyze_resource_trends()
        if trends.get('cpu_avg', 0) > 80:
            recommendations.append("High CPU usage detected - consider load balancing or upgrading hardware")
        if trends.get('memory_avg', 0) > 85:
            recommendations.append("High memory usage - implement memory optimization or increase RAM")
        if trends.get('gpu_memory_avg', 0) > 90:
            recommendations.append("GPU memory nearly full - use model quantization or offloading")
        return recommendations

    def schedule_auto_report(self, interval_sec: int = 600):
        """Launch a background task to log performance reports periodically."""
        async def reporter():
            while True:
                report = self.get_performance_report()
                logger.info(f"Automated Performance Report: {json.dumps(report, default=str)}")
                await asyncio.sleep(interval_sec)
        asyncio.create_task(reporter())

# ==================== ADVANCED CACHING SYSTEM ====================

class IntelligentCache:
    """Multi-level caching with TTL and intelligent eviction"""

    def __init__(self, max_memory_mb: int = 2048):
        self.memory_cache = {}
        self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0, 'evictions': 0})
        self.access_times = {}
        self.cache_sizes = {}
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_memory = 0

    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Fallback to 1KB

    async def get(self, key: str, generator: Callable = None, ttl: int = 3600):
        """Get from cache or generate if missing"""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if datetime.now() < entry['expires']:
                self.cache_stats[key]['hits'] += 1
                self.access_times[key] = datetime.now()
                return entry['value']
            else:
                await self.evict(key)
        self.cache_stats[key]['misses'] += 1
        if generator:
            value = await generator() if asyncio.iscoroutinefunction(generator) else generator()
            await self.set(key, value, ttl)
            return value
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cache value with TTL"""
        size = self._estimate_size(value)
        while self.current_memory + size > self.max_memory and self.memory_cache:
            await self._evict_lru()
        self.memory_cache[key] = {
            'value': value,
            'expires': datetime.now() + timedelta(seconds=ttl),
            'size': size
        }
        self.cache_sizes[key] = size
        self.current_memory += size
        self.access_times[key] = datetime.now()

    async def evict(self, key: str):
        """Evict specific key from cache"""
        if key in self.memory_cache:
            size = self.cache_sizes.get(key, 0)
            del self.memory_cache[key]
            del self.cache_sizes[key]
            del self.access_times[key]
            self.current_memory -= size
            self.cache_stats[key]['evictions'] += 1

    async def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        await self.evict(lru_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(s['hits'] for s in self.cache_stats.values())
        total_misses = sum(s['misses'] for s in self.cache_stats.values())
        return {
            'memory_used_mb': self.current_memory / (1024 * 1024),
            'memory_limit_mb': self.max_memory / (1024 * 1024),
            'items_cached': len(self.memory_cache),
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
            'top_cached': sorted(
                [(k, v['hits']) for k, v in self.cache_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    async def clear_all(self):
        """Full cache flush, with GC"""
        self.memory_cache.clear()
        self.cache_sizes.clear()
        self.access_times.clear()
        self.current_memory = 0
        gc.collect()

# ==================== ENHANCED TASK QUEUE ====================

@dataclass
class Task:
    id: str
    type: str
    content: Any
    priority: int = 5
    model_hint: Optional[str] = None
    requires_vision: bool = False
    complexity: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat())

class PriorityTaskQueue:
    """Advanced task queue with dynamic priority adjustment"""

    def __init__(self):
        self.queue = []
        self.task_map = {}
        self.priority_adjustments = defaultdict(float)
        self.task_dependencies = defaultdict(set)
        self.completed_tasks = set()
        self.task_metrics = defaultdict(lambda: {'wait_time': 0, 'execution_time': 0})

    async def submit(self, task: Task, dependencies: List[str] = None) -> str:
        adjusted_priority = task.priority + self.priority_adjustments.get(task.type, 0)
        self.task_map[task.id] = {
            'task': task,
            'submitted': datetime.now(),
            'adjusted_priority': adjusted_priority,
            'dependencies': set(dependencies) if dependencies else set()
        }
        if not dependencies or all(dep in self.completed_tasks for dep in dependencies):
            heapq.heappush(self.queue, (-adjusted_priority, datetime.now(), task.id))
        else:
            for dep in dependencies:
                self.task_dependencies[dep].add(task.id)
        return task.id

    async def get_next(self) -> Optional[Tuple[Task, Dict[str, Any]]]:
        while self.queue:
            _, submitted_time, task_id = heapq.heappop(self.queue)
            if task_id in self.task_map:
                task_info = self.task_map[task_id]
                task = task_info['task']
                wait_time = (datetime.now() - task_info['submitted']).total_seconds()
                self.task_metrics[task.type]['wait_time'] = (
                    self.task_metrics[task.type]['wait_time'] * 0.9 + wait_time * 0.1
                )
                return task, task_info
        return None, None

    async def complete(self, task_id: str, execution_time: float):
        if task_id not in self.task_map:
            return
        task_info = self.task_map[task_id]
        task_type = task_info['task'].type
        self.task_metrics[task_type]['execution_time'] = (
            self.task_metrics[task_type]['execution_time'] * 0.9 + execution_time * 0.1
        )
        self.completed_tasks.add(task_id)
        del self.task_map[task_id]
        if task_id in self.task_dependencies:
            for dependent_id in self.task_dependencies[task_id]:
                if dependent_id in self.task_map:
                    dep_info = self.task_map[dependent_id]
                    dep_info['dependencies'].discard(task_id)
                    if not dep_info['dependencies']:
                        heapq.heappush(
                            self.queue,
                            (-dep_info['adjusted_priority'], datetime.now(), dependent_id)
                        )
            del self.task_dependencies[task_id]

    def adjust_priorities(self):
        for task_type, metrics in self.task_metrics.items():
            if metrics['wait_time'] > 10:
                self.priority_adjustments[task_type] += 0.1
            elif metrics['wait_time'] < 2:
                self.priority_adjustments[task_type] -= 0.05
            self.priority_adjustments[task_type] = max(-2, min(2, self.priority_adjustments[task_type]))

# ==================== ENHANCED MEMORY SYSTEM ====================

class MemorySystem:
    """Base memory system for extension (placeholder for compatibility)."""
    def __init__(self, persist_directory: str = "./lexos_memory"):
        pass

class HierarchicalMemorySystem(MemorySystem):
    """Advanced memory with hierarchical storage and intelligent retrieval"""

    def __init__(self, persist_directory: str = "./lexos_memory"):
        super().__init__(persist_directory)
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory / "chroma")
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="lexos_memories",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer('all-mpnet-base-v2')  # Upgraded embedding model
        self.semantic_clusters = defaultdict(list)
        self.memory_graph = nx.Graph()
        self.importance_scores = {}
        self.memory_decay_rates = {}
        self.access_frequency = defaultdict(int)
        self.long_term_collection = self.chroma_client.get_or_create_collection(
            name="lexos_long_term",
            metadata={"hnsw:space": "cosine"}
        )

    async def store_memory(self, content: str, metadata: Dict[str, Any]) -> str:
        # Deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = self.collection.get(where={'content_hash': content_hash})
        if existing and existing['ids']:
            logger.info("Duplicate memory detected, skipping storage.")
            return existing['ids'][0]
        memory_id = hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        embedding = self.embedder.encode(content)
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[{**metadata, 'content_hash': content_hash}]
        )
        importance = self._calculate_importance(content, metadata)
        self.importance_scores[memory_id] = importance
        self.memory_decay_rates[memory_id] = 0.01 / (importance + 0.1)
        cluster_id = self._find_semantic_cluster(embedding)
        self.semantic_clusters[cluster_id].append(memory_id)
        await self._update_memory_connections(memory_id, embedding)
        return memory_id

    def _calculate_importance(self, content: str, metadata: Dict[str, Any]) -> float:
        importance = 0.5
        if metadata.get('task_type') == 'reasoning':
            importance += 0.2
        if metadata.get('success', False):
            importance += 0.1
        if 'error' in metadata:
            importance -= 0.2
        if len(content) > 500:
            importance += 0.1
        if any(keyword in content.lower() for keyword in ['important', 'critical', 'key']):
            importance += 0.15
        return max(0.1, min(1.0, importance))

    def _find_semantic_cluster(self, embedding: np.ndarray) -> int:
        if not self.semantic_clusters:
            return 0
        min_distance = float('inf')
        best_cluster = 0
        for cluster_id, memory_ids in self.semantic_clusters.items():
            if memory_ids:
                cluster_embeddings = []
                for mem_id in memory_ids[:10]:
                    result = self.collection.get(ids=[mem_id], include=['embeddings'])
                    if result['embeddings']:
                        cluster_embeddings.append(result['embeddings'][0])
                if cluster_embeddings:
                    centroid = np.mean(cluster_embeddings, axis=0)
                    distance = np.linalg.norm(embedding - centroid)
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_id
        if min_distance > 0.5:
            best_cluster = len(self.semantic_clusters)
        return best_cluster

    async def _update_memory_connections(self, memory_id: str, embedding: np.ndarray):
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=5,
            where={"$ne": {"id": memory_id}}
        )
        self.memory_graph.add_node(memory_id)
        for i, related_id in enumerate(results['ids'][0]):
            if results['distances'][0][i] < 0.3:
                self.memory_graph.add_edge(
                    memory_id,
                    related_id,
                    weight=1 - results['distances'][0][i]
                )

    async def recall_memories(self, query: str, n_results: int = 5, strategy: str = "hybrid") -> List[Dict]:
        self.access_frequency[query] += 1
        if strategy == "semantic":
            return await self._recall_semantic(query, n_results)
        elif strategy == "importance":
            return await self._recall_by_importance(query, n_results)
        elif strategy == "graph":
            return await self._recall_by_graph(query, n_results)
        else:
            return await self._recall_hybrid(query, n_results)

    async def _recall_semantic(self, query: str, n_results: int) -> List[Dict]:
        embedding = self.embedder.encode(query)
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results
        )
        memories = []
        for i in range(len(results['ids'][0])):
            memories.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        return memories

    async def _recall_by_importance(self, query: str, n_results: int) -> List[Dict]:
        candidates = await self._recall_semantic(query, n_results * 3)
        for memory in candidates:
            memory_id = memory['id']
            importance = self.importance_scores.get(memory_id, 0.5)
            memory['combined_score'] = (1 - memory['distance']) * 0.7 + importance * 0.3
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates[:n_results]

    async def _recall_by_graph(self, query: str, n_results: int) -> List[Dict]:
        seeds = await self._recall_semantic(query, 2)
        if not seeds:
            return []
        related_memories = set()
        for seed in seeds:
            seed_id = seed['id']
            if seed_id in self.memory_graph:
                neighbors = list(self.memory_graph.neighbors(seed_id))
                related_memories.update(neighbors[:n_results//2])
        results = []
        for memory_id in list(related_memories)[:n_results]:
            memory_data = self.collection.get(ids=[memory_id])
            if memory_data['documents']:
                results.append({
                    'id': memory_id,
                    'content': memory_data['documents'][0],
                    'metadata': memory_data['metadatas'][0] if memory_data['metadatas'] else {},
                    'retrieval_method': 'graph'
                })
        return results

    async def _recall_hybrid(self, query: str, n_results: int) -> List[Dict]:
        semantic_results = await self._recall_semantic(query, n_results)
        importance_results = await self._recall_by_importance(query, n_results//2)
        graph_results = await self._recall_by_graph(query, n_results//2)
        all_results = {}
        for result in semantic_results + importance_results + graph_results:
            memory_id = result['id']
            if memory_id not in all_results:
                all_results[memory_id] = result
            else:
                if result.get('distance', 1) < all_results[memory_id].get('distance', 1):
                    all_results[memory_id] = result
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get('distance', 1)
        )
        return sorted_results[:n_results]

    async def memory_maintenance(self):
        current_time = datetime.now()
        memories_to_archive = []
        for memory_id, decay_rate in self.memory_decay_rates.items():
            if memory_id in self.importance_scores:
                self.importance_scores[memory_id] *= (1 - decay_rate)
                if self.importance_scores[memory_id] < 0.1:
                    memories_to_archive.append(memory_id)
        for memory_id in memories_to_archive:
            await self._archive_memory(memory_id)

    async def _archive_memory(self, memory_id: str):
        memory_data = self.collection.get(ids=[memory_id])
        if memory_data['documents']:
            self.long_term_collection.add(
                ids=[memory_id],
                embeddings=memory_data['embeddings'][0] if memory_data['embeddings'] else None,
                documents=memory_data['documents'][0],
                metadatas=[{
                    **memory_data['metadatas'][0],
                    'archived_at': datetime.now().isoformat()
                }]
            )
            self.collection.delete(ids=[memory_id])
            if memory_id in self.importance_scores:
                del self.importance_scores[memory_id]
            if memory_id in self.memory_decay_rates:
                del self.memory_decay_rates[memory_id]
            if memory_id in self.memory_graph:
                self.memory_graph.remove_node(memory_id]

# ==================== ADVANCED ERROR HANDLING ====================

class ErrorRecoverySystem:
    """Sophisticated error handling and recovery"""

    def __init__(self):
        self.error_patterns = defaultdict(lambda: {'count': 0, 'last_seen': None, 'recovery_strategies': []})
        self.recovery_success_rate = defaultdict(float)
        self.circuit_breakers = {}

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        error_type = type(error).__name__
        error_key = f"{error_type}:{str(error)[:100]}"
        self.error_patterns[error_key]['count'] += 1
        self.error_patterns[error_key]['last_seen'] = datetime.now()
        if self._is_circuit_open(error_key):
            logger.warning(f"Circuit breaker open for {error_key}")
            return None
        recovery_strategies = self._get_recovery_strategies(error, context)
        for strategy in recovery_strategies:
            try:
                logger.info(f"Attempting recovery strategy: {strategy['name']}")
                result = await strategy['handler'](error, context)
                self.recovery_success_rate[strategy['name']] = (
                    self.recovery_success_rate[strategy['name']] * 0.9 + 0.1
                )
                return result
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy['name']} failed: {recovery_error}")
                self.recovery_success_rate[strategy['name']] *= 0.9
        if self.error_patterns[error_key]['count'] > 10:
            self._open_circuit(error_key)
        return None

    def _get_recovery_strategies(self, error: Exception, context: Dict[str, Any]) -> List[Dict]:
        strategies = []
        if isinstance(error, Exception) and "model" in str(error).lower():
            strategies.append({
                'name': 'fallback_model',
                'handler': self._fallback_model_strategy
            })
            strategies.append({
                'name': 'retry_with_delay',
                'handler': self._retry_strategy
            })
        if isinstance(error, MemoryError) or "memory" in str(error).lower():
            strategies.append({
                'name': 'clear_cache',
                'handler': self._clear_cache_strategy
            })
            strategies.append({
                'name': 'reduce_batch_size',
                'handler': self._reduce_batch_strategy
            })
        if isinstance(error, aiohttp.ClientError) or "connection" in str(error).lower():
            strategies.append({
                'name': 'exponential_backoff',
                'handler': self._exponential_backoff_strategy
            })
        strategies.sort(
            key=lambda s: self.recovery_success_rate.get(s['name'], 0.5),
            reverse=True
        )
        return strategies

    async def _fallback_model_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        if 'task' in context and 'orchestrator' in context:
            task = context['task']
            orchestrator = context['orchestrator']
            fallback_models = {
                'llama3.3:70b-instruct-q4_K_M': 'llama3.2:3b',
                'qwen2.5-coder:32b': 'phi3:mini',
                'llava:34b': 'llava:7b',
                'deepseek-r1:7b': 'mistral:7b'
            }
            original_model = context.get('model')
            if original_model in fallback_models:
                fallback = fallback_models[original_model]
                logger.info(f"Falling back from {original_model} to {fallback}")
                return await orchestrator.ollama.generate(
                    model=fallback,
                    prompt=task.content,
                    options={'num_predict': 512}
                )
        raise error

    async def _retry_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        max_retries = 3
        for i in range(max_retries):
            await asyncio.sleep(2 ** i)
            try:
                if 'original_func' in context:
                    return await context['original_func']()
            except Exception as e:
                if i == max_retries - 1:
                    raise

    async def _clear_cache_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        if 'cache' in context:
            cache = context['cache']
            logger.info("Clearing cache to free memory")
            await cache.clear_all()
        gc.collect()
        if 'original_func' in context:
            return await context['original_func']()

    async def _reduce_batch_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        if 'batch_size' in context:
            new_batch_size = max(1, context['batch_size'] // 2)
            logger.info(f"Reducing batch size from {context['batch_size']} to {new_batch_size}")
            context['batch_size'] = new_batch_size
            if 'original_func' in context:
                return await context['original_func']()

    async def _exponential_backoff_strategy(self, error: Exception, context: Dict[str, Any]) -> Any:
        max_retries = 5
        base_delay = 1
        for i in range(max_retries):
            delay = base_delay * (2 ** i) + random.uniform(0, 1)
            logger.info(f"Retrying after {delay:.1f} seconds (attempt {i+1}/{max_retries})")
            await asyncio.sleep(delay)
            try:
                if 'original_func' in context:
                    return await context['original_func']()
            except Exception as e:
                if i == max_retries - 1:
                    raise

    def _is_circuit_open(self, error_key: str) -> bool:
        if error_key in self.circuit_breakers:
            breaker = self.circuit_breakers[error_key]
            if datetime.now() < breaker['reset_time']:
                return True
            else:
                del self.circuit_breakers[error_key]
        return False

    def _open_circuit(self, error_key: str, duration_minutes: int = 5):
        self.circuit_breakers[error_key] = {
            'opened_at': datetime.now(),
            'reset_time': datetime.now() + timedelta(minutes=duration_minutes)
        }
        logger.warning(f"Circuit breaker opened for {error_key} for {duration_minutes} minutes")

# ==================== DISTRIBUTED PROCESSING ====================

class DistributedTaskProcessor:
    """Handle distributed processing across multiple nodes"""

    def __init__(self, node_id: str = "main"):
        self.node_id = node_id
        self.worker_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.remote_nodes = {}
        self.load_balancer = LoadBalancer()

    async def register_node(self, node_url: str, capabilities: Dict[str, Any]):
        self.remote_nodes[node_url] = {
            'url': node_url,
            'capabilities': capabilities,
            'status': 'active',
            'load': 0,
            'last_heartbeat': datetime.now()
        }

    async def distribute_task(self, task: Task) -> Dict[str, Any]:
        best_node = self.load_balancer.select_node(
            task,
            self.remote_nodes,
            local_load=self._get_local_load()
        )
        if best_node == 'local':
            return await self._process_locally(task)
        else:
            return await self._process_remotely(task, best_node)

    async def _process_locally(self, task: Task) -> Dict[str, Any]:
        if task.type in ['reasoning', 'analysis']:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.worker_pool,
                self._cpu_intensive_task,
                task
            )
        else:
            return {'processed_by': self.node_id, 'result': 'processed'}

    async def _process_remotely(self, task: Task, node_url: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{node_url}/process",
                    json={'task': task.__dict__},
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    result = await response.json()
                    self.remote_nodes[node_url]['load'] -= 1
                    return result
            except Exception as e:
                logger.error(f"Remote processing failed on {node_url}: {e}")
                self.remote_nodes[node_url]['status'] = 'error'
                return await self._process_locally(task)

    def _get_local_load(self) -> float:
        cpu_load = psutil.cpu_percent() / 100
        memory_load = psutil.virtual_memory().percent / 100
        gpu_load = 0
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_load = max(g.load for g in gpus)
        return (cpu_load * 0.4 + memory_load * 0.3 + gpu_load * 0.3)

    def _cpu_intensive_task(self, task: Task) -> Dict[str, Any]:
        time.sleep(0.1)
        return {'processed': True, 'node': self.node_id}

class LoadBalancer:
    """Intelligent load balancing across nodes"""

    def __init__(self):
        self.strategy = 'weighted_round_robin'
        self.node_weights = defaultdict(float)

    def select_node(self, task: Task, nodes: Dict[str, Any], local_load: float) -> str:
        if not nodes or all(n['status'] != 'active' for n in nodes.values()):
            return 'local'
        active_nodes = {
            url: info for url, info in nodes.items()
            if info['status'] == 'active'
        }
        if task.requires_vision:
            capable_nodes = {
                url: info for url, info in active_nodes.items()
                if info['capabilities'].get('vision', False)
            }
            if not capable_nodes:
                return 'local'
            active_nodes = capable_nodes
        if self.strategy == 'least_loaded':
            return self._select_least_loaded(active_nodes, local_load)
        elif self.strategy == 'weighted_round_robin':
            return self._select_weighted_round_robin(active_nodes, local_load)
        else:
            return self._select_random(active_nodes, local_load)

    def _select_least_loaded(self, nodes: Dict[str, Any], local_load: float) -> str:
        node_loads = {url: info['load'] for url, info in nodes.items()}
        node_loads['local'] = local_load * 10
        return min(node_loads.items(), key=lambda x: x[1])[0]

    def _select_weighted_round_robin(self, nodes: Dict[str, Any], local_load: float) -> str:
        weights = {}
        for url, info in nodes.items():
            load = info['load'] + 1
            weights[url] = 1.0 / load
        weights['local'] = 1.0 / (local_load * 10 + 1)
        total_weight = sum(weights.values())
        rand = random.uniform(0, total_weight)
        current = 0
        for node, weight in weights.items():
            current += weight
            if rand <= current:
                return node
        return 'local'

    def _select_random(self, nodes: Dict[str, Any], local_load: float) -> str:
        all_nodes = list(nodes.keys()) + ['local']
        return random.choice(all_nodes)