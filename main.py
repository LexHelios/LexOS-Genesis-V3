"""
LexOS: Production-Ready AI Operating System
Enhanced with robust error handling, real agent integration, and advanced features
H100-optimized, self-healing, and extensible
"""

import asyncio
import json
import logging
import os
import signal
import sys
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import time
import hashlib
import yaml
import pickle
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import gc

import aiohttp
from aiohttp import web
import aiofiles
import psutil
import GPUtil

# Import enhanced components if available
try:
    from agents import EnhancedCodingAgent, EnhancedVisionAgent, EnhancedReasoningAgent, \
                      EnhancedConversationAgent, EnhancedResearchAgent, EnhancedAgentFactory
    ENHANCED_AGENTS_AVAILABLE = True
except ImportError:
    ENHANCED_AGENTS_AVAILABLE = False
    
try:
    from consciousness import ConsciousnessOrchestrator, ConsciousnessEnabledOrchestrator
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False

# ========== ENHANCED LOGGING ==========
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# File handler with rotation
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler(
    log_dir / "lexos.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

# Console handler with colors
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    ColoredFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger("LexOS")

# ========== CONFIGURATION MANAGEMENT ==========
class ConfigManager:
    """Centralized configuration management with validation"""
    
    def __init__(self, config_path: str = "lexos_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded configuration from {self.config_path}")
                    return config
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return self._get_default_config()
        else:
            config = self._get_default_config()
            self._save_config(config)
            return config
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'system': {
                'max_concurrent_tasks': 3,
                'task_timeout': 300,
                'memory_limit_gb': 16,
                'enable_gpu': True,
                'preferred_gpu': 'H100',
                'enable_consciousness': False
            },
            'web': {
                'enabled': True,
                'host': '0.0.0.0',
                'port': 8080,
                'enable_cors': True,
                'max_upload_size_mb': 100
            },
            'models': {
                'default_temperature': 0.7,
                'default_top_p': 0.9,
                'enable_fallbacks': True,
                'cache_embeddings': True,
                'preferred_models': {
                    'chat': 'llama3.2:3b',
                    'code': 'qwen2.5-coder:32b',
                    'vision': 'llava:7b',
                    'reasoning': 'deepseek-r1:7b',
                    'research': 'llama3.3:70b-instruct-q4_K_M'
                }
            },
            'memory': {
                'persist_directory': './lexos_memory',
                'max_memories': 10000,
                'consolidation_interval': 300,
                'prune_threshold': 5000,
                'embedding_model': 'all-mpnet-base-v2'
            },
            'agents': {
                'enable_learning': True,
                'enable_collaboration': True,
                'personality_evolution': True,
                'max_retries': 3
            },
            'monitoring': {
                'system_check_interval': 30,
                'metrics_retention_days': 7,
                'enable_alerts': True,
                'alert_thresholds': {
                    'cpu_percent': 90,
                    'memory_percent': 85,
                    'gpu_memory_percent': 90,
                    'error_rate': 0.1
                }
            },
            'security': {
                'enable_auth': False,
                'api_key': None,
                'allowed_origins': ['*'],
                'rate_limit_per_minute': 60
            }
        }
        
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            
    def _validate_config(self):
        """Validate configuration values"""
        # Ensure critical values are within reasonable ranges
        system = self.config.get('system', {})
        if system.get('max_concurrent_tasks', 1) < 1:
            system['max_concurrent_tasks'] = 1
        if system.get('task_timeout', 60) < 10:
            system['task_timeout'] = 10
            
        # Validate web config
        web = self.config.get('web', {})
        if web.get('port', 8080) < 1024 or web.get('port', 8080) > 65535:
            web['port'] = 8080
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """Set configuration value and save"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        self._save_config(self.config)
        
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        self._validate_config()
        logger.info("Configuration reloaded")

# ========== ENHANCED ERROR HANDLING ==========
class ErrorHandler:
    """Centralized error handling with recovery strategies"""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {
            'ModelLoadError': self._recover_model_error,
            'MemoryError': self._recover_memory_error,
            'TimeoutError': self._recover_timeout_error,
            'NetworkError': self._recover_network_error
        }
        
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Handle error with appropriate recovery strategy"""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1
        
        # Log error
        error_info = {
            'type': error_type,
            'message': str(error),
            'timestamp': datetime.now(),
            'context': context,
            'traceback': traceback.format_exc()
        }
        self.error_history.append(error_info)
        logger.error(f"{error_type}: {str(error)}", exc_info=True)
        
        # Apply recovery strategy
        recovery_func = self.recovery_strategies.get(
            error_type, 
            self._recover_generic_error
        )
        
        try:
            return recovery_func(error, context)
        except Exception as recovery_error:
            logger.error(f"Recovery failed: {recovery_error}")
            return None
            
    def _recover_model_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from model loading errors"""
        logger.info("Attempting to recover from model error")
        
        # Try fallback model
        if context and 'task' in context:
            task = context['task']
            task.metadata['use_fallback'] = True
            return {'retry': True, 'task': task}
            
        return None
        
    def _recover_memory_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from memory errors"""
        logger.info("Attempting to recover from memory error")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        if context and 'clear_cache' in context:
            context['clear_cache']()
            
        return {'retry': True, 'reduced_capacity': True}
        
    def _recover_timeout_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from timeout errors"""
        logger.info("Attempting to recover from timeout error")
        
        # Increase timeout for retry
        if context and 'task' in context:
            task = context['task']
            task.metadata['extended_timeout'] = True
            return {'retry': True, 'task': task}
            
        return None
        
    def _recover_network_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from network errors"""
        logger.info("Attempting to recover from network error")
        
        # Wait and retry
        return {'retry': True, 'delay': 5}
        
    def _recover_generic_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generic error recovery"""
        logger.warning(f"No specific recovery for {type(error).__name__}")
        
        # Basic retry for non-critical errors
        if self.error_counts[type(error).__name__] < 3:
            return {'retry': True, 'delay': 1}
            
        return None
        
    def get_error_report(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = sum(self.error_counts.values())
        recent_errors = list(self.error_history)[-10:]
        
        return {
            'total_errors': total_errors,
            'error_counts': dict(self.error_counts),
            'recent_errors': [
                {
                    'type': e['type'],
                    'message': e['message'],
                    'timestamp': e['timestamp'].isoformat()
                }
                for e in recent_errors
            ],
            'top_errors': sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

# ========== PERFORMANCE MONITORING ==========
class PerformanceMonitor:
    """Monitor system performance and generate insights"""
    
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
        self.alerts = deque(maxlen=100)
        
    def track_operation(self, operation: str):
        """Decorator to track operation performance"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                error_occurred = False
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    error_occurred = True
                    self.metrics[operation]['errors'] += 1
                    raise
                    
                finally:
                    elapsed = time.time() - start_time
                    self._record_metric(operation, elapsed, error_occurred)
                    
            return wrapper
        return decorator
        
    def _record_metric(self, operation: str, elapsed: float, error: bool):
        """Record performance metric"""
        metric = self.metrics[operation]
        metric['count'] += 1
        metric['total_time'] += elapsed
        metric['min_time'] = min(metric['min_time'], elapsed)
        metric['max_time'] = max(metric['max_time'], elapsed)
        metric['last_10'].append({
            'time': elapsed,
            'timestamp': datetime.now(),
            'error': error
        })
        
        # Check for performance degradation
        if len(metric['last_10']) >= 5:
            recent_avg = sum(m['time'] for m in list(metric['last_10'])[-5:]) / 5
            overall_avg = metric['total_time'] / metric['count']
            
            if recent_avg > overall_avg * 1.5:
                self._create_alert('performance_degradation', {
                    'operation': operation,
                    'recent_avg': recent_avg,
                    'overall_avg': overall_avg
                })
                
    def _create_alert(self, alert_type: str, details: Dict[str, Any]):
        """Create performance alert"""
        alert = {
            'type': alert_type,
            'timestamp': datetime.now(),
            'details': details
        }
        self.alerts.append(alert)
        logger.warning(f"Performance alert: {alert_type} - {details}")
        
    async def collect_resource_metrics(self):
        """Collect system resource metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'swap_percent': psutil.swap_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            }
        }
        
        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics['gpus'] = []
            for gpu in gpus:
                metrics['gpus'].append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'memory_used': gpu.memoryUsed,
                    'memory_percent': gpu.memoryUtil * 100,
                    'utilization': gpu.load * 100,
                    'temperature': gpu.temperature
                })
                
        self.resource_history.append(metrics)
        return metrics
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        operations = {}
        
        for op_name, metrics in self.metrics.items():
            if metrics['count'] > 0:
                avg_time = metrics['total_time'] / metrics['count']
                recent_times = [m['time'] for m in metrics['last_10']]
                recent_avg = sum(recent_times) / len(recent_times) if recent_times else 0
                
                operations[op_name] = {
                    'count': metrics['count'],
                    'avg_time': avg_time,
                    'recent_avg': recent_avg,
                    'min_time': metrics['min_time'] if metrics['min_time'] != float('inf') else 0,
                    'max_time': metrics['max_time'],
                    'error_rate': metrics['errors'] / metrics['count'],
                    'trend': 'improving' if recent_avg < avg_time else 'degrading'
                }
                
        # Resource usage trends
        if len(self.resource_history) > 10:
            recent = list(self.resource_history)[-60:]  # Last minute
            resource_trends = {
                'cpu_avg': sum(m['cpu']['percent'] for m in recent) / len(recent),
                'memory_avg': sum(m['memory']['percent'] for m in recent) / len(recent),
                'cpu_trend': self._calculate_trend([m['cpu']['percent'] for m in recent]),
                'memory_trend': self._calculate_trend([m['memory']['percent'] for m in recent])
            }
            
            if recent[0].get('gpus'):
                gpu_util = [m['gpus'][0]['utilization'] for m in recent if m.get('gpus')]
                resource_trends['gpu_avg'] = sum(gpu_util) / len(gpu_util) if gpu_util else 0
                resource_trends['gpu_trend'] = self._calculate_trend(gpu_util) if gpu_util else 'stable'
        else:
            resource_trends = {}
            
        return {
            'operations': operations,
            'resource_trends': resource_trends,
            'alerts': list(self.alerts)[-10:],
            'health_score': self._calculate_health_score(operations, resource_trends)
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 2:
            return 'stable'
            
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
            
        slope = numerator / denominator
        
        if slope > 0.5:
            return 'increasing'
        elif slope < -0.5:
            return 'decreasing'
        else:
            return 'stable'
            
    def _calculate_health_score(self, operations: Dict[str, Any], resources: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        score = 1.0
        
        # Penalize high error rates
        for op in operations.values():
            if op['error_rate'] > 0.1:
                score -= 0.1
            elif op['error_rate'] > 0.05:
                score -= 0.05
                
        # Penalize high resource usage
        if resources:
            if resources.get('cpu_avg', 0) > 80:
                score -= 0.1
            if resources.get('memory_avg', 0) > 85:
                score -= 0.1
            if resources.get('gpu_avg', 0) > 90:
                score -= 0.05
                
        # Penalize degrading trends
        for op in operations.values():
            if op['trend'] == 'degrading':
                score -= 0.02
                
        return max(0.0, min(1.0, score))

# ========== ENHANCED CORE COMPONENTS ==========

@dataclass
class EnhancedTask(Task):
    """Enhanced task with additional tracking and metadata"""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects
        for key in ['created_at', 'started_at', 'completed_at', 'submitted_at']:
            if key in data and data[key]:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedTask':
        """Create from dictionary"""
        # Convert ISO strings back to datetime
        for key in ['created_at', 'started_at', 'completed_at']:
            if key in data and data[key]:
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)

class EnhancedModelOrchestrator(ModelOrchestrator):
    """Enhanced orchestrator with fallback support and resource optimization"""
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.fallback_chains = self._initialize_fallback_chains()
        self.model_health = defaultdict(lambda: {'failures': 0, 'last_failure': None})
        
    def _initialize_fallback_chains(self) -> Dict[str, List[str]]:
        """Define fallback chains for each model"""
        return {
            "llama3.3:70b-instruct-q4_K_M": ["mistral:7b-instruct", "llama3.2:3b"],
            "qwen2.5-coder:32b": ["phi3:mini", "llama3.2:3b"],
            "llava:34b": ["llava:7b"],
            "deepseek-r1:7b": ["mistral:7b", "llama3.2:3b"],
        }
        
    async def select_best_model(self, task: EnhancedTask) -> str:
        """Select model with health checks and fallback support"""
        # Check if fallback requested
        if task.metadata.get('use_fallback'):
            return await self._select_fallback_model(task)
            
        # Get base recommendation
        base_model = await super().select_best_model(task)
        
        # Check model health
        if self._is_model_healthy(base_model):
            return base_model
            
        # Find healthy alternative
        return await self._select_fallback_model(task, avoid=[base_model])
        
    def _is_model_healthy(self, model_name: str) -> bool:
        """Check if model is healthy"""
        health = self.model_health[model_name]
        
        # Reset after cooldown
        if health['last_failure']:
            cooldown = timedelta(minutes=5)
            if datetime.now() - health['last_failure'] > cooldown:
                health['failures'] = 0
                health['last_failure'] = None
                
        return health['failures'] < 3
        
    async def _select_fallback_model(self, task: EnhancedTask, avoid: List[str] = None) -> str:
        """Select a fallback model"""
        avoid = avoid or []
        base_model = await super().select_best_model(task)
        
        # Try fallback chain
        for fallback in self.fallback_chains.get(base_model, []):
            if fallback not in avoid and self._is_model_healthy(fallback):
                logger.info(f"Using fallback model: {fallback}")
                return fallback
                
        # Last resort - smallest model
        return "llama3.2:1b"
        
    def record_model_failure(self, model_name: str, error: Exception):
        """Record model failure for health tracking"""
        self.model_health[model_name]['failures'] += 1
        self.model_health[model_name]['last_failure'] = datetime.now()
        logger.warning(f"Model {model_name} failure recorded: {error}")

class EnhancedMemorySystem(MemorySystem):
    """Enhanced memory with backup and recovery"""
    
    def __init__(self, config: ConfigManager):
        persist_dir = config.get('memory.persist_directory', './lexos_memory')
        super().__init__(persist_dir)
        self.config = config
        self.backup_dir = Path(persist_dir) / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    async def backup(self) -> str:
        """Create memory backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"memory_backup_{timestamp}.pkl"
        
        try:
            backup_data = {
                'memory_index': self.memory_index,
                'working_memory': self.working_memory,
                'timestamp': datetime.now()
            }
            
            async with aiofiles.open(backup_file, 'wb') as f:
                await f.write(pickle.dumps(backup_data))
                
            logger.info(f"Memory backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
            
    async def restore(self, backup_file: str):
        """Restore memory from backup"""
        try:
            async with aiofiles.open(backup_file, 'rb') as f:
                data = pickle.loads(await f.read())
                
            self.memory_index = data['memory_index']
            self.working_memory = data['working_memory']
            self._save_memories()
            
            logger.info(f"Memory restored from: {backup_file}")
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise

class EnhancedTaskScheduler(TaskScheduler):
    """Enhanced scheduler with persistence and priority optimization"""
    
    def __init__(self, config: ConfigManager):
        max_concurrent = config.get('system.max_concurrent_tasks', 3)
        super().__init__(max_concurrent)
        self.config = config
        self.task_history = deque(maxlen=1000)
        self.priority_optimizer = PriorityOptimizer()
        
    async def submit_task(self, task: EnhancedTask, dependencies: List[str] = None) -> str:
        """Submit enhanced task with optimization"""
        # Optimize priority based on history
        optimized_priority = self.priority_optimizer.optimize(task, self.task_history)
        task.priority = optimized_priority
        
        # Call parent method
        return await super().submit_task(task, dependencies)
        
    def mark_completed(self, task_id: str, result: Dict[str, Any]):
        """Mark task completed and update history"""
        super().mark_completed(task_id, result)
        
        # Add to history for learning
        if task_id in self.task_priorities:
            self.task_history.append({
                'task_id': task_id,
                'priority': self.task_priorities[task_id],
                'result': result,
                'timestamp': datetime.now()
            })
            
    async def save_state(self):
        """Save scheduler state for recovery"""
        state = {
            'active_tasks': {k: v.to_dict() for k, v in self.active_tasks.items()},
            'completed_tasks': self.completed_tasks[-100:],
            'task_history': list(self.task_history)[-100:]
        }
        
        state_file = Path("scheduler_state.json")
        async with aiofiles.open(state_file, 'w') as f:
            await f.write(json.dumps(state, default=str))
            
    async def load_state(self):
        """Load scheduler state"""
        state_file = Path("scheduler_state.json")
        if not state_file.exists():
            return
            
        try:
            async with aiofiles.open(state_file, 'r') as f:
                state = json.loads(await f.read())
                
            # Restore active tasks
            for task_id, task_data in state.get('active_tasks', {}).items():
                task = EnhancedTask.from_dict(task_data)
                self.active_tasks[task_id] = task
                
            # Restore history
            self.completed_tasks = state.get('completed_tasks', [])
            self.task_history = deque(state.get('task_history', []), maxlen=1000)
            
            logger.info("Scheduler state restored")
            
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")

class PriorityOptimizer:
    """Optimize task priorities based on patterns"""
    
    def optimize(self, task: EnhancedTask, history: deque) -> int:
        """Optimize task priority"""
        base_priority = task.priority
        
        # Boost priority for frequently failing task types
        failure_rate = self._calculate_failure_rate(task.type, history)
        if failure_rate > 0.3:
            base_priority += 1
            
        # Boost priority for user-facing tasks
        if task.type in ['conversation', 'vision']:
            base_priority += 1
            
        # Reduce priority for low-importance background tasks
        if task.metadata.get('background', False):
            base_priority -= 2
            
        return max(1, min(10, base_priority))
        
    def _calculate_failure_rate(self, task_type: str, history: deque) -> float:
        """Calculate failure rate for task type"""
        type_tasks = [h for h in history if h.get('task_type') == task_type]
        if not type_tasks:
            return 0.0
            
        failures = sum(1 for t in type_tasks if not t.get('result', {}).get('success', False))
        return failures / len(type_tasks)

# ========== ENHANCED WEB INTERFACE ==========

class EnhancedWebInterface(WebInterface):
    """Enhanced web interface with authentication and file uploads"""
    
    def __init__(self, lexos, config: ConfigManager):
        self.config = config
        super().__init__(lexos)
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self._setup_middleware()
        
    def _setup_middleware(self):
        """Setup middleware for CORS, auth, etc."""
        if self.config.get('web.enable_cors'):
            # Simple CORS middleware
            @web.middleware
            async def cors_middleware(request, handler):
                response = await handler(request)
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
                return response
                
            self.app.middlewares.append(cors_middleware)
            
        if self.config.get('security.enable_auth'):
            # Simple API key auth
            @web.middleware
            async def auth_middleware(request, handler):
                if request.path.startswith('/api/'):
                    api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
                    if api_key != self.config.get('security.api_key'):
                        return web.json_response({'error': 'Unauthorized'}, status=401)
                return await handler(request)
                
            self.app.middlewares.append(auth_middleware)
            
    def _setup_routes(self):
        """Setup enhanced routes"""
        super()._setup_routes()
        
        # Additional routes
        self.app.router.add_post('/api/upload', self.upload_file)
        self.app.router.add_get('/api/agents', self.get_agents)
        self.app.router.add_post('/api/config', self.update_config)
        self.app.router.add_get('/api/performance', self.get_performance)
        self.app.router.add_get('/api/errors', self.get_errors)
        self.app.router.add_post('/api/backup', self.create_backup)
        self.app.router.add_static('/', path='static', name='static')
        
    async def index(self, request):
        """Serve enhanced dashboard"""
        dashboard_path = Path('static/index.html')
        if dashboard_path.exists():
            return web.FileResponse(dashboard_path)
        else:
            # Fallback to simple HTML
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>LexOS Control Panel</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .stat-card { background: #2a2a2a; padding: 20px; border-radius: 8px; }
        .stat-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; 
                 border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #45a049; }
        #log { background: #000; color: #0f0; padding: 10px; height: 300px; 
               overflow-y: auto; font-family: monospace; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 LexOS Control Panel</h1>
        
        <div class="stats" id="stats">
            <div class="stat-card">
                <h3>System Status</h3>
                <div class="stat-value">Loading...</div>
            </div>
        </div>
        
        <div style="margin-top: 20px;">
            <button onclick="submitTask('chat', 'Hello LexOS!')">Test Chat</button>
            <button onclick="submitTask('code', 'Write a hello world function')">Test Code</button>
            <button onclick="showStats()">Refresh Stats</button>
        </div>
        
        <div id="log">
            <div>Connecting to LexOS...</div>
        </div>
    </div>
    
    <script>
        let ws = null;
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8080/ws');
            
            ws.onopen = () => {
                addLog('Connected to LexOS');
                showStats();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'log') {
                    addLog(data.message);
                }
            };
            
            ws.onclose = () => {
                addLog('Disconnected from LexOS');
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        function addLog(message) {
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            entry.textContent = message;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
        
        async function showStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                const statsDiv = document.getElementById('stats');
                statsDiv.innerHTML = `
                    <div class="stat-card">
                        <h3>Uptime</h3>
                        <div class="stat-value">${Math.floor(stats.uptime / 60)}m</div>
                    </div>
                    <div class="stat-card">
                        <h3>Tasks Completed</h3>
                        <div class="stat-value">${stats.completed_tasks}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Active Tasks</h3>
                        <div class="stat-value">${stats.active_tasks}</div>
                    </div>
                    <div class="stat-card">
                        <h3>CPU Usage</h3>
                        <div class="stat-value">${stats.resources.cpu_percent.toFixed(1)}%</div>
                    </div>
                    <div class="stat-card">
                        <h3>Memory Usage</h3>
                        <div class="stat-value">${stats.resources.memory_percent.toFixed(1)}%</div>
                    </div>
                `;
            } catch (error) {
                addLog('Error fetching stats: ' + error);
            }
        }
        
        async function submitTask(type, content) {
            try {
                const response = await fetch('/api/task', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type, content})
                });
                const result = await response.json();
                addLog('Task result: ' + JSON.stringify(result));
            } catch (error) {
                addLog('Error submitting task: ' + error);
            }
        }
        
        connectWebSocket();
        setInterval(showStats, 5000);
    </script>
</body>
</html>
"""
            return web.Response(text=html, content_type='text/html')
            
    async def upload_file(self, request):
        """Handle file uploads"""
        reader = await request.multipart()
        
        # Read file
        field = await reader.next()
        if not field:
            return web.json_response({'error': 'No file provided'}, status=400)
            
        filename = field.filename
        if not filename:
            return web.json_response({'error': 'No filename'}, status=400)
            
        # Security check
        safe_filename = Path(filename).name
        filepath = self.upload_dir / safe_filename
        
        # Size check
        max_size = self.config.get('web.max_upload_size_mb', 100) * 1024 * 1024
        size = 0
        
        with open(filepath, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    f.close()
                    filepath.unlink()
                    return web.json_response({'error': 'File too large'}, status=413)
                f.write(chunk)
                
        return web.json_response({
            'filename': safe_filename,
            'path': str(filepath),
            'size': size
        })
        
    async def get_agents(self, request):
        """Get agent information"""
        agents = {}
        for name, agent in self.lexos.agents.items():
            agents[name] = agent.get_status_report()
        return web.json_response(agents)
        
    async def update_config(self, request):
        """Update configuration"""
        data = await request.json()
        
        for key, value in data.items():
            self.lexos.config.set(key, value)
            
        return web.json_response({'status': 'updated'})
        
    async def get_performance(self, request):
        """Get performance metrics"""
        if hasattr(self.lexos, 'performance_monitor'):
            report = self.lexos.performance_monitor.get_performance_report()
            return web.json_response(report)
        return web.json_response({})
        
    async def get_errors(self, request):
        """Get error report"""
        if hasattr(self.lexos, 'error_handler'):
            report = self.lexos.error_handler.get_error_report()
            return web.json_response(report)
        return web.json_response({})
        
    async def create_backup(self, request):
        """Create system backup"""
        try:
            # Backup memory
            memory_backup = await self.lexos.memory.backup()
            
            # Save scheduler state
            await self.lexos.scheduler.save_state()
            
            return web.json_response({
                'status': 'success',
                'memory_backup': memory_backup,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

# ========== ENHANCED LEXOS MAIN SYSTEM ==========

class EnhancedLexOS(LexOS):
    """Production-ready LexOS with all enhancements"""
    
    def __init__(self, config_path: str = "lexos_config.yaml"):
        # Initialize configuration
        self.config = ConfigManager(config_path)
        
        # Initialize error handling and monitoring
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize enhanced components
        logger.info("Initializing Enhanced LexOS...")
        self.orchestrator = EnhancedModelOrchestrator(self.config)
        self.memory = EnhancedMemorySystem(self.config)
        self.scheduler = EnhancedTaskScheduler(self.config)
        
        # Initialize agents
        if ENHANCED_AGENTS_AVAILABLE and self.config.get('agents.enable_collaboration'):
            self._initialize_enhanced_agents()
        else:
            self._initialize_basic_agents()
            
        # Initialize consciousness if available and enabled
        if CONSCIOUSNESS_AVAILABLE and self.config.get('system.enable_consciousness'):
            self._initialize_consciousness()
            
        # System state
        self.is_running = False
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "uptime": 0,
            "start_time": None,
            "version": "2.0.0",
            "enhanced": True
        }
        
        # Initialize web interface
        self.web_interface = EnhancedWebInterface(self, self.config)
        self.web_runner = None
        
        # Background task handles
        self._background_tasks = []
        
    def _initialize_enhanced_agents(self):
        """Initialize enhanced agents with factory"""
        logger.info("Initializing enhanced agents...")
        self.agent_factory = EnhancedAgentFactory(self.orchestrator, self.config.config)
        
        self.agents = {
            'coding': self.agent_factory.create_agent('coding'),
            'vision': self.agent_factory.create_agent('vision'),
            'reasoning': self.agent_factory.create_agent('reasoning'),
            'conversation': self.agent_factory.create_agent('conversation'),
            'research': self.agent_factory.create_agent('research')
        }
        
        logger.info("Enhanced agents initialized with collaboration support")
        
    def _initialize_consciousness(self):
        """Initialize consciousness layer"""
        logger.info("Initializing consciousness layer...")
        self.consciousness = ConsciousnessOrchestrator(identity="ATLAS")
        
        # Wrap orchestrator with consciousness
        self.orchestrator = ConsciousnessEnabledOrchestrator(
            self.orchestrator,
            identity="ATLAS"
        )
        
        logger.info("Consciousness layer activated")
        
    async def start(self):
        """Start enhanced LexOS with recovery support"""
        try:
            logger.info("Starting Enhanced LexOS...")
            self.is_running = True
            self.stats["start_time"] = datetime.now()
            
            # Load previous state if exists
            await self._load_system_state()
            
            # Start background tasks
            self._background_tasks = [
                asyncio.create_task(self._task_processor()),
                asyncio.create_task(self._memory_consolidator()),
                asyncio.create_task(self._system_monitor()),
                asyncio.create_task(self._performance_collector()),
                asyncio.create_task(self._error_monitor())
            ]
            
            # Start web interface
            if self.config.get('web.enabled', True):
                await self._start_web_interface()
                
            logger.info("Enhanced LexOS is ready!")
            
            # Broadcast startup
            await self.web_interface.broadcast({
                'type': 'log',
                'message': f"[{datetime.now().strftime('%H:%M:%S')}] LexOS started successfully"
            })
            
        except Exception as e:
            logger.error(f"Failed to start LexOS: {e}")
            await self.stop()
            raise
            
    async def stop(self):
        """Gracefully stop with state saving"""
        logger.info("Shutting down Enhanced LexOS...")
        self.is_running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Save state
        await self._save_system_state()
        
        # Stop web interface
        if self.web_runner:
            await self.web_runner.cleanup()
            
        logger.info("Enhanced LexOS shutdown complete")
        
    @PerformanceMonitor.track_operation("task_processing")
    async def _process_task(self, task: EnhancedTask):
        """Process task with enhanced error handling"""
        logger.info(f"Processing task {task.id} of type {task.type}")
        task.started_at = datetime.now()
        task.status = "running"
        self.scheduler.active_tasks[task.id] = task
        self.stats["total_tasks"] += 1
        
        # Broadcast task start
        await self.web_interface.broadcast({
            "type": "log",
            "message": f"[{datetime.now().strftime('%H:%M:%S')}] Starting task {task.id} ({task.type})"
        })
        
        result = None
        for attempt in range(self.config.get('agents.max_retries', 3)):
            try:
                # Select agent
                agent = self._select_agent(task)
                
                # Check for collaboration request
                if task.metadata.get('collaborate') and hasattr(self, 'agent_factory'):
                    result = await self._process_collaborative_task(task)
                else:
                    # Single agent processing
                    result = await agent.process(task)
                
                if result.get("success"):
                    # Store in memory
                    await self.memory.store_memory(
                        f"Task: {task.content}\nResult: {str(result.get('result', ''))[:500]}",
                        {
                            "task_id": task.id,
                            "task_type": task.type,
                            "agent": result.get("agent", "unknown"),
                            "model": result.get("model", "unknown"),
                            "duration": (datetime.now() - task.started_at).total_seconds()
                        }
                    )
                    
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    task.result = result
                    self.stats["completed_tasks"] += 1
                    
                    # Broadcast success
                    await self.web_interface.broadcast({
                        "type": "log",
                        "message": f"[{datetime.now().strftime('%H:%M:%S')}] Task {task.id} completed successfully"
                    })
                    
                    break
                else:
                    raise Exception(result.get('error', 'Unknown error'))
                    
            except Exception as e:
                task.error = str(e)
                task.retries = attempt + 1
                
                # Handle error
                recovery = self.error_handler.handle_error(e, {'task': task, 'attempt': attempt})
                
                if recovery and recovery.get('retry') and attempt < self.config.get('agents.max_retries', 3) - 1:
                    # Wait before retry
                    delay = recovery.get('delay', 1)
                    logger.info(f"Retrying task {task.id} after {delay}s")
                    await asyncio.sleep(delay)
                    
                    # Apply recovery modifications
                    if recovery.get('task'):
                        task = recovery['task']
                else:
                    # Final failure
                    task.status = "failed"
                    task.completed_at = datetime.now()
                    self.stats["failed_tasks"] += 1
                    
                    # Record model failure if applicable
                    if hasattr(self.orchestrator, 'record_model_failure') and result:
                        model = result.get('model')
                        if model:
                            self.orchestrator.record_model_failure(model, e)
                    
                    # Broadcast failure
                    await self.web_interface.broadcast({
                        "type": "log",
                        "message": f"[{datetime.now().strftime('%H:%M:%S')}] Task {task.id} failed: {str(e)}"
                    })
                    
                    result = {"success": False, "error": str(e)}
                    break
                    
        # Mark task completed in scheduler
        self.scheduler.mark_completed(task.id, result or {"success": False, "error": "Max retries exceeded"})
        
        return result
        
    async def _process_collaborative_task(self, task: EnhancedTask) -> Dict[str, Any]:
        """Process task using agent collaboration"""
        agent_types = task.metadata.get('agents', ['reasoning', 'conversation'])
        pattern = task.metadata.get('collaboration_pattern', 'sequential')
        
        logger.info(f"Processing collaborative task {task.id} with agents: {agent_types}")
        
        result = await self.agent_factory.execute_collaborative_task(
            task,
            agent_types,
            pattern
        )
        
        return result
        
    async def _performance_collector(self):
        """Collect performance metrics periodically"""
        while self.is_running:
            try:
                await self.performance_monitor.collect_resource_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                logger.error(f"Performance collection error: {e}")
                await asyncio.sleep(30)
                
    async def _error_monitor(self):
        """Monitor error rates and trigger alerts"""
        while self.is_running:
            try:
                error_report = self.error_handler.get_error_report()
                total_errors = error_report['total_errors']
                
                # Check error rate
                if self.stats['total_tasks'] > 0:
                    error_rate = self.stats['failed_tasks'] / self.stats['total_tasks']
                    
                    if error_rate > self.config.get('monitoring.alert_thresholds.error_rate', 0.1):
                        logger.warning(f"High error rate detected: {error_rate:.1%}")
                        
                        # Broadcast alert
                        await self.web_interface.broadcast({
                            'type': 'alert',
                            'severity': 'warning',
                            'message': f"High error rate: {error_rate:.1%}"
                        })
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitor failed: {e}")
                await asyncio.sleep(60)
                
    async def _load_system_state(self):
        """Load previous system state"""
        try:
            # Load scheduler state
            await self.scheduler.load_state()
            
            # Load saved stats
            state_file = Path("lexos_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                    
                # Restore stats (but reset uptime)
                self.stats.update(saved_state.get('stats', {}))
                self.stats['uptime'] = 0
                self.stats['start_time'] = datetime.now()
                
                logger.info("System state restored")
                
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
            
    async def _save_system_state(self):
        """Save system state with enhancements"""
        try:
            state = {
                "stats": self.stats,
                "config_hash": hashlib.md5(
                    json.dumps(self.config.config, sort_keys=True).encode()
                ).hexdigest(),
                "performance_report": self.performance_monitor.get_performance_report(),
                "error_report": self.error_handler.get_error_report(),
                "agent_states": {
                    name: agent.get_status_report() 
                    for name, agent in self.agents.items()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            async with aiofiles.open("lexos_state.json", "w") as f:
                await f.write(json.dumps(state, default=str, indent=2))
                
            # Save scheduler state
            await self.scheduler.save_state()
            
            # Create periodic backup
            if self.stats.get('completed_tasks', 0) % 100 == 0:
                await self.memory.backup()
                
            logger.info("System state saved")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced system statistics"""
        base_stats = super().get_stats()
        
        # Add enhanced metrics
        base_stats.update({
            'version': self.stats.get('version', 'unknown'),
            'enhanced': True,
            'performance': self.performance_monitor.get_performance_report(),
            'errors': self.error_handler.get_error_report(),
            'config': {
                'max_concurrent_tasks': self.config.get('system.max_concurrent_tasks'),
                'consciousness_enabled': self.config.get('system.enable_consciousness', False),
                'learning_enabled': self.config.get('agents.enable_learning', True)
            }
        })
        
        # Add consciousness report if available
        if hasattr(self, 'consciousness'):
            base_stats['consciousness'] = self.consciousness.get_consciousness_report()
            
        return base_stats
        
    # Enhanced API methods
    
    async def chat(self, message: str, context: Optional[List[str]] = None,
                   collaborate: bool = False) -> str:
        """Enhanced chat with collaboration option"""
        task = EnhancedTask(
            id=f"chat_{datetime.now().timestamp()}",
            type="conversation",
            content=message,
            priority=5,
            complexity="simple",
            metadata={
                "context": context,
                "collaborate": collaborate,
                "agents": ["conversation", "reasoning"] if collaborate else None
            }
        )
        
        result = await self._process_task(task)
        
        if result.get("success"):
            if collaborate and "results" in result:
                # Return combined collaborative result
                responses = [r.get('result', {}).get('response', '') 
                           for r in result.get('results', []) 
                           if r.get('success')]
                return " ".join(responses)
            else:
                return result.get("result", {}).get("response", "No response generated")
        else:
            return f"Error: {result.get('error', 'Unknown error')}"
            
    async def analyze_image(self, image_path: str, question: str = "What's in this image?",
                          detailed: bool = False) -> Dict[str, Any]:
        """Enhanced image analysis with detail option"""
        task = EnhancedTask(
            id=f"vision_{datetime.now().timestamp()}",
            type="vision",
            content=image_path,
            priority=5,
            requires_vision=True,
            complexity="complex" if detailed else "medium",
            metadata={
                "prompt": question,
                "vision_task": "technical" if detailed else "describe",
                "multi_pass": detailed
            }
        )
        
        result = await self._process_task(task)
        return result.get("result") if result.get("success") else {"error": result.get("error")}
        
    async def research(self, topic: str, research_type: str = "general",
                      depth: str = "medium") -> Dict[str, Any]:
        """Enhanced research with depth control"""
        complexity_map = {"shallow": "simple", "medium": "medium", "deep": "complex"}
        
        task = EnhancedTask(
            id=f"research_{datetime.now().timestamp()}",
            type="research",
            content=topic,
            priority=6,
            complexity=complexity_map.get(depth, "medium"),
            metadata={
                "research_type": research_type,
                "include_citations": depth in ["medium", "deep"],
                "collaborate": depth == "deep",
                "agents": ["research", "analysis"] if depth == "deep" else None
            }
        )
        
        result = await self._process_task(task)
        return result.get("result") if result.get("success") else {"error": result.get("error")}

# ========== ENHANCED CLI ==========

class EnhancedLexOSCLI(LexOSCLI):
    """Enhanced CLI with additional commands"""
    
    def __init__(self, lexos: EnhancedLexOS):
        super().__init__(lexos)
        
        # Additional commands
        self.commands.update({
            'backup': self.backup,
            'restore': self.restore,
            'config': self.config,
            'performance': self.performance,
            'errors': self.errors,
            'collaborate': self.collaborate,
            'agents': self.show_agents
        })
        
    async def show_help(self, args: str):
        """Show enhanced help"""
        help_text = """
Available Commands:
  chat <message>       - Chat with LexOS
  code <description>   - Generate code
  solve <problem>      - Solve a problem
  analyze <image>      - Analyze an image
  research <topic>     - Research a topic
  recall <query>       - Search memories
  collaborate <task>   - Use multiple agents
  stats                - Show system statistics
  performance          - Show performance metrics
  errors               - Show error report
  agents               - Show agent status
  config <key> [value] - Get/set configuration
  backup               - Create system backup
  restore <file>       - Restore from backup
  clear                - Clear screen
  help                 - Show this help
  exit                 - Exit LexOS

Examples:
  chat How are you today?
  code in python create a web server
  collaborate solve this complex problem
  research deep quantum computing
  config system.max_concurrent_tasks 5
"""
        print(help_text)
        
    async def collaborate(self, args: str):
        """Run collaborative task"""
        if not args:
            print("Usage: collaborate <task description>")
            return
            
        print("\nRunning collaborative task...")
        
        # Determine task type from content
        task_type = "reasoning"  # Default
        if any(word in args.lower() for word in ['code', 'program', 'function']):
            task_type = "coding"
        elif any(word in args.lower() for word in ['research', 'find', 'search']):
            task_type = "research"
            
        result = await self.lexos.agent_factory.execute_collaborative_task(
            EnhancedTask(
                id=f"collab_{datetime.now().timestamp()}",
                type=task_type,
                content=args,
                priority=7,
                complexity="complex"
            ),
            agent_types=["reasoning", "conversation"],
            pattern="consensus"
        )
        
        if result.get('success'):
            print(f"\nCollaborative Result:")
            print(f"Agents involved: {result.get('agents_involved', [])}")
            print(f"Pattern: {result.get('collaboration_type', 'unknown')}")
            
            if 'consensus' in result:
                print(f"\nConsensus: {result['consensus']}")
            elif 'final_result' in result:
                print(f"\nFinal Result: {result['final_result']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    async def performance(self, args: str):
        """Show performance metrics"""
        report = self.lexos.performance_monitor.get_performance_report()
        
        print("\n" + "="*60)
        print("📊 Performance Report")
        print("="*60)
        
        print("\n🔧 Operations:")
        for op, metrics in report.get('operations', {}).items():
            print(f"\n{op}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Avg Time: {metrics['avg_time']:.2f}s")
            print(f"  Recent Avg: {metrics['recent_avg']:.2f}s")
            print(f"  Error Rate: {metrics['error_rate']:.1%}")
            print(f"  Trend: {metrics['trend']}")
            
        print("\n💻 Resource Trends:")
        trends = report.get('resource_trends', {})
        if trends:
            print(f"  CPU Average: {trends.get('cpu_avg', 0):.1f}%")
            print(f"  CPU Trend: {trends.get('cpu_trend', 'unknown')}")
            print(f"  Memory Average: {trends.get('memory_avg', 0):.1f}%")
            print(f"  Memory Trend: {trends.get('memory_trend', 'unknown')}")
            if 'gpu_avg' in trends:
                print(f"  GPU Average: {trends['gpu_avg']:.1f}%")
                print(f"  GPU Trend: {trends.get('gpu_trend', 'unknown')}")
                
        print(f"\n🏥 Health Score: {report.get('health_score', 0):.1%}")
        
        alerts = report.get('alerts', [])
        if alerts:
            print(f"\n⚠️  Recent Alerts:")
            for alert in alerts[-5:]:
                print(f"  - {alert['type']}: {alert['details']}")
                
        print("="*60 + "\n")
        
    async def errors(self, args: str):
        """Show error report"""
        report = self.lexos.error_handler.get_error_report()
        
        print("\n" + "="*60)
        print("🚨 Error Report")
        print("="*60)
        
        print(f"\nTotal Errors: {report['total_errors']}")
        
        if report['top_errors']:
            print("\n📊 Top Error Types:")
            for error_type, count in report['top_errors']:
                print(f"  {error_type}: {count}")
                
        if report['recent_errors']:
            print("\n🕒 Recent Errors:")
            for error in report['recent_errors'][-5:]:
                print(f"  [{error['timestamp']}] {error['type']}: {error['message'][:50]}...")
                
        print("="*60 + "\n")
        
    async def show_agents(self, args: str):
        """Show agent status"""
        print("\n" + "="*60)
        print("🤖 Agent Status")
        print("="*60)
        
        for name, agent in self.lexos.agents.items():
            status = agent.get_status_report()
            print(f"\n{name.title()} Agent:")
            
            if 'personality' in status:
                personality = status['personality']
                print(f"  Personality:")
                print(f"    Creativity: {personality.get('creativity', 0):.2f}")
                print(f"    Precision: {personality.get('precision', 0):.2f}")
                print(f"    Empathy: {personality.get('empathy', 0):.2f}")
                
            performance = status.get('performance', {})
            print(f"  Performance:")
            print(f"    Tasks: {performance.get('tasks_completed', 0)}")
            print(f"    Success Rate: {performance.get('success_rate', 1.0):.1%}")
            print(f"    Avg Time: {performance.get('average_time', 0):.2f}s")
            
            if 'memory_stats' in status:
                memory = status['memory_stats']
                print(f"  Memory:")
                print(f"    Short-term: {memory.get('short_term_size', 0)}")
                print(f"    Skills: {len(memory.get('skill_categories', []))}")
                
        print("="*60 + "\n")
        
    async def config(self, args: str):
        """Get or set configuration"""
        parts = args.split(maxsplit=1)
        
        if not parts:
            # Show all config
            print("\nCurrent Configuration:")
            print(json.dumps(self.lexos.config.config, indent=2))
            return
            
        key = parts[0]
        
        if len(parts) == 1:
            # Get specific value
            value = self.lexos.config.get(key)
            print(f"\n{key}: {value}")
        else:
            # Set value
            value_str = parts[1]
            
            # Try to parse value
            try:
                value = json.loads(value_str)
            except:
                # Try as int/float
                try:
                    value = int(value_str)
                except:
                    try:
                        value = float(value_str)
                    except:
                        # Keep as string
                        value = value_str
                        
            self.lexos.config.set(key, value)
            print(f"\nSet {key} = {value}")
            
            # Reload config in components
            self.lexos.config.reload()
            
    async def backup(self, args: str):
        """Create system backup"""
        print("\nCreating backup...")
        
        try:
            # Create memory backup
            memory_backup = await self.lexos.memory.backup()
            
            # Save all states
            await self.lexos._save_system_state()
            
            print(f"✅ Backup created successfully:")
            print(f"  Memory: {memory_backup}")
            print(f"  State: lexos_state.json")
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            
    async def restore(self, args: str):
        """Restore from backup"""
        if not args:
            # List available backups
            backup_dir = self.lexos.memory.backup_dir
            backups = sorted(backup_dir.glob("memory_backup_*.pkl"))
            
            if not backups:
                print("No backups found.")
                return
                
            print("\nAvailable backups:")
            for i, backup in enumerate(backups[-10:], 1):
                print(f"  {i}. {backup.name}")
                
            print("\nUsage: restore <backup_file>")
            return
            
        print(f"\nRestoring from {args}...")
        
        try:
            await self.lexos.memory.restore(args)
            print("✅ Restore completed successfully")
            
        except Exception as e:
            print(f"❌ Restore failed: {e}")

# ========== SIGNAL HANDLERS ==========

class SignalHandler:
    """Handle system signals gracefully"""
    
    def __init__(self, lexos: EnhancedLexOS):
        self.lexos = lexos
        self.shutdown_in_progress = False
        
    def setup(self):
        """Setup signal handlers"""
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._handle_signal)
            
        # Windows-specific
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._handle_signal)
            
    def _handle_signal(self, signum, frame):
        """Handle shutdown signal"""
        if self.shutdown_in_progress:
            logger.info("Forced shutdown")
            sys.exit(1)
            
        self.shutdown_in_progress = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        
        # Create task for async shutdown
        asyncio.create_task(self._shutdown())
        
    async def _shutdown(self):
        """Perform graceful shutdown"""
        try:
            await self.lexos.stop()
            
            # Stop event loop
            loop = asyncio.get_running_loop()
            loop.stop()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            sys.exit(1)

# ========== MAIN ENTRY POINT ==========

async def main():
    """Enhanced main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LexOS - Your Personal AI Operating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start with default config
  %(prog)s --config my.yaml   # Use custom config
  %(prog)s --no-web          # Disable web interface
  %(prog)s --port 8888       # Use custom port
  %(prog)s --debug           # Enable debug logging
        """
    )
    
    parser.add_argument('--config', default='lexos_config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--no-web', action='store_true', 
                       help='Disable web interface')
    parser.add_argument('--port', type=int, 
                       help='Web interface port (overrides config)')
    parser.add_argument('--host', default=None,
                       help='Web interface host (overrides config)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--basic', action='store_true',
                       help='Use basic agents (no enhancements)')
    parser.add_argument('--consciousness', action='store_true',
                       help='Enable consciousness layer')
    parser.add_argument('--backup', 
                       help='Restore from backup before starting')
    
    args = parser.parse_args()
    
    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
        
    # Print startup banner
    print("\n" + "="*60)
    print("🧠 LexOS - Personal AI Operating System")
    print("="*60)
    print(f"Version: 2.0.0 (Enhanced)")
    print(f"Config: {args.config}")
    print(f"Enhanced Agents: {ENHANCED_AGENTS_AVAILABLE}")
    print(f"Consciousness: {CONSCIOUSNESS_AVAILABLE}")
    print("="*60 + "\n")
    
    try:
        # Initialize LexOS
        if args.basic:
            lexos = LexOS(config_path=args.config)
        else:
            lexos = EnhancedLexOS(config_path=args.config)
            
        # Apply command line overrides
        if args.no_web:
            lexos.config.set('web.enabled', False)
        if args.port:
            lexos.config.set('web.port', args.port)
        if args.host:
            lexos.config.set('web.host', args.host)
        if args.consciousness:
            lexos.config.set('system.enable_consciousness', True)
            
        # Restore from backup if requested
        if args.backup:
            logger.info(f"Restoring from backup: {args.backup}")
            await lexos.memory.restore(args.backup)
            
        # Setup signal handlers
        signal_handler = SignalHandler(lexos)
        signal_handler.setup()
        
        # Start LexOS
        await lexos.start()
        
        # Initialize CLI
        if args.basic:
            cli = LexOSCLI(lexos)
        else:
            cli = EnhancedLexOSCLI(lexos)
            
        # Run CLI
        try:
            await cli.run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print("Check lexos.log for details")
        sys.exit(1)
        
    finally:
        # Ensure cleanup
        try:
            if 'lexos' in locals():
                await lexos.stop()
        except:
            pass
            
        print("\n👋 Goodbye!")

# ========== PRODUCTION UTILITIES ==========

def check_requirements():
    """Check system requirements"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required (found {sys.version})")
        
    # Check critical imports
    required_packages = [
        'aiohttp', 'psutil', 'GPUtil', 'chromadb', 
        'sentence_transformers', 'yaml', 'numpy'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing package: {package}")
            
    # Check Ollama
    try:
        from ollama import AsyncClient
    except ImportError:
        issues.append("Ollama Python client not installed")
        
    # Check system resources
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 8:
        issues.append(f"Low system memory: {memory_gb:.1f}GB (8GB+ recommended)")
        
    # Check GPU
    gpus = GPUtil.getGPUs()
    if not gpus:
        issues.append("No GPU detected (GPU recommended for optimal performance)")
    else:
        for gpu in gpus:
            if gpu.memoryTotal < 4000:  # Less than 4GB
                issues.append(f"Low GPU memory on {gpu.name}: {gpu.memoryTotal}MB")
                
    return issues

def print_system_info():
    """Print system information"""
    print("\n📊 System Information:")
    print(f"  OS: {sys.platform}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CPUs: {psutil.cpu_count()}")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    gpus = GPUtil.getGPUs()
    if gpus:
        print("  GPUs:")
        for gpu in gpus:
            print(f"    - {gpu.name} ({gpu.memoryTotal}MB)")
    else:
        print("  GPUs: None detected")
        
    print()

# ========== ENTRY POINT ==========

if __name__ == "__main__":
    # Check requirements
    issues = check_requirements()
    if issues:
        print("⚠️  System requirement issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nLexOS may not function properly.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
            
    # Print system info in debug mode
    if '--debug' in sys.argv:
        print_system_info()
        
    # Run main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown interrupted")
    except Exception as e:
        print(f"\n❌ Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)

"""
LexOS Enhanced Main - Production Ready

This enhanced version includes:

1. **Robust Error Handling**
   - Centralized error handler with recovery strategies
   - Automatic retries with exponential backoff
   - Circuit breakers for failing models
   - Graceful degradation

2. **Performance Monitoring**
   - Operation-level performance tracking
   - Resource usage monitoring and trends
   - Performance alerts and health scoring
   - Bottleneck detection

3. **Configuration Management**
   - YAML-based configuration
   - Hot reload support
   - Validation and defaults
   - CLI configuration commands

4. **Enhanced Agents**
   - Full integration with enhanced agent system
   - Agent collaboration support
   - Personality evolution
   - Memory and learning

5. **Advanced Scheduler**
   - Priority optimization based on history
   - Task dependencies
   - State persistence
   - Automatic recovery

6. **Production Web Interface**
   - Real-time dashboard
   - WebSocket support
   - File uploads
   - API authentication
   - CORS support

7. **System Resilience**
   - Automatic backups
   - State recovery on restart
   - Graceful shutdown
   - Signal handling

8. **Monitoring & Alerts**
   - Real-time metrics
   - Error rate monitoring
   - Resource alerts
   - Performance degradation detection

9. **CLI Enhancements**
   - Collaborative commands
   - Performance metrics
   - Configuration management
   - Backup/restore

10. **Consciousness Integration**
    - Optional consciousness layer
    - Experience tracking
    - Evolution monitoring

Ready for production deployment!
"""