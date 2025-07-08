"""
LexOS Core Brain - Core Architecture & Model Management
Optimized for your Ollama model collection and personal H100
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
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
import gc
import time
import networkx as nx

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("lexos_core.log", mode='a')
    ]
)
logger = logging.getLogger("LexOS")

@dataclass
class ModelProfile:
    """Profile for each model's capabilities and optimal use cases"""
    name: str
    size_gb: float
    capabilities: List[str]
    best_for: List[str]
    speed: str  # fast, medium, slow
    context_window: int = 4096
    quantization: Optional[str] = None
    updated_at: Optional[str] = None

@dataclass
class Task:
    """Represents a task to be processed by LexOS"""
    id: str
    type: str
    content: Any
    priority: int = 5
    model_hint: Optional[str] = None
    requires_vision: bool = False
    complexity: str = "medium"  # simple, medium, complex
    metadata: Dict[str, Any] = field(default_factory=dict)
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat())

class ModelOrchestrator:
    """Intelligently routes tasks to the best model based on requirements, H100 aware."""

    def __init__(self):
        self.ollama = OllamaClient()
        self.model_profiles = self._initialize_model_profiles()
        self.model_usage_stats = {}  # model_name: {calls, avg_latency, errors}
        self.active_models = set()
        self.model_gpu_assignments = {}  # model_name: gpu_id
        self.preferred_speed = None
        self._detect_gpus()

    def _detect_gpus(self):
        gpus = GPUtil.getGPUs()
        logger.info(f"GPUs detected: {[f'{g.id}: {g.name}, {g.memoryFree:.1f}MB free' for g in gpus]}")
        if not gpus:
            logger.warning("No GPU detected, running in CPU mode.")
        else:
            for g in gpus:
                if "H100" in g.name.upper():
                    logger.info(f"Detected NVIDIA H100: {g.name} (ID {g.id}) - will prioritize for big models.")

    def _initialize_model_profiles(self) -> Dict[str, ModelProfile]:
        return {
            # Vision Models
            "llava:7b": ModelProfile(
                name="llava:7b",
                size_gb=4.7,
                capabilities=["vision", "image_understanding", "visual_qa"],
                best_for=["image_analysis", "screenshot_reading", "visual_tasks"],
                speed="medium"
            ),
            "llava:34b": ModelProfile(
                name="llava:34b",
                size_gb=20,
                capabilities=["vision", "advanced_image_understanding", "detailed_visual_analysis"],
                best_for=["complex_image_analysis", "medical_images", "technical_diagrams"],
                speed="slow"
            ),
            "qwen2.5-coder:32b": ModelProfile(
                name="qwen2.5-coder:32b",
                size_gb=19,
                capabilities=["coding", "debugging", "code_review", "architecture"],
                best_for=["code_generation", "bug_fixing", "code_explanation", "refactoring"],
                speed="slow",
                context_window=32768
            ),
            "deepseek-r1:7b": ModelProfile(
                name="deepseek-r1:7b",
                size_gb=4.7,
                capabilities=["reasoning", "problem_solving", "mathematics", "logic"],
                best_for=["complex_reasoning", "math_problems", "logical_puzzles", "analysis"],
                speed="medium"
            ),
            "llama3.3:70b-instruct-q4_K_M": ModelProfile(
                name="llama3.3:70b-instruct-q4_K_M",
                size_gb=42,
                capabilities=["advanced_reasoning", "creative_writing", "complex_analysis", "research"],
                best_for=["complex_tasks", "long_form_content", "detailed_analysis", "creative_work"],
                speed="slow",
                context_window=8192
            ),
            "mistral:7b-instruct": ModelProfile(
                name="mistral:7b-instruct",
                size_gb=4.1,
                capabilities=["instruction_following", "general_tasks", "conversation"],
                best_for=["general_qa", "task_execution", "quick_responses"],
                speed="fast"
            ),
            "mistral:7b": ModelProfile(
                name="mistral:7b",
                size_gb=4.1,
                capabilities=["general_text", "completion", "basic_reasoning"],
                best_for=["text_generation", "simple_tasks", "fast_responses"],
                speed="fast"
            ),
            "llama3.2:1b": ModelProfile(
                name="llama3.2:1b",
                size_gb=1.3,
                capabilities=["basic_tasks", "simple_qa", "quick_responses"],
                best_for=["simple_queries", "basic_chat", "high_volume_tasks"],
                speed="very_fast"
            ),
            "llama3.2:3b": ModelProfile(
                name="llama3.2:3b",
                size_gb=2.0,
                capabilities=["balanced_tasks", "general_qa", "moderate_reasoning"],
                best_for=["general_chat", "moderate_complexity", "good_speed_quality_balance"],
                speed="fast"
            ),
            "phi3:mini": ModelProfile(
                name="phi3:mini",
                size_gb=2.2,
                capabilities=["efficient_reasoning", "basic_coding", "general_tasks"],
                best_for=["quick_reasoning", "simple_code", "efficient_processing"],
                speed="fast"
            )
        }

    async def select_best_model(self, task: Task) -> str:
        # Vision tasks
        if task.requires_vision:
            if task.complexity == "complex":
                return "llava:34b"
            return "llava:7b"
        # Coding
        if task.type == "coding":
            if task.complexity == "complex":
                return "qwen2.5-coder:32b"
            candidates = [k for k, v in self.model_profiles.items() if "coding" in v.capabilities and v.speed in ("fast", "very_fast")]
            if candidates:
                return candidates[0]
            return "phi3:mini"
        # Reasoning & Math
        if task.type in ("reasoning", "math"):
            if task.complexity == "complex":
                return "llama3.3:70b-instruct-q4_K_M"
            return "deepseek-r1:7b"
        # General
        if task.complexity == "complex":
            return "llama3.3:70b-instruct-q4_K_M"
        elif task.complexity == "simple":
            return "llama3.2:1b"
        else:
            return "llama3.2:3b"

    async def check_model_availability(self, model_name: str) -> bool:
        try:
            gpu_available = len(GPUtil.getGPUs()) > 0
            model_size = self.model_profiles[model_name].size_gb * 1024  # MB
            if gpu_available:
                # Prefer H100
                h100_gpus = [g for g in GPUtil.getGPUs() if "H100" in g.name.upper()]
                gpu = h100_gpus[0] if h100_gpus else GPUtil.getGPUs()[0]
                free_memory = gpu.memoryFree
                if free_memory < model_size:
                    logger.warning(f"Insufficient GPU memory for {model_name} on {gpu.name}: {free_memory:.1f}MB < {model_size:.1f}MB")
                    return False
            else:
                ram = psutil.virtual_memory()
                if (ram.available / (1024 ** 2)) < model_size:
                    logger.warning(f"Insufficient system RAM for {model_name}: {ram.available / (1024 ** 2):.1f}MB < {model_size:.1f}MB")
                    return False
            # Ollama test
            await self.ollama.generate(
                model=model_name,
                prompt="test",
                options={"num_predict": 1}
            )
            return True
        except Exception as e:
            logger.error(f"Model {model_name} not available: {e}")
            return False

    async def load_model(self, model_name: str) -> bool:
        """Load model, assign to best GPU and mark as active."""
        if model_name in self.active_models:
            return True
        try:
            gpus = GPUtil.getGPUs()
            h100_gpus = [g for g in gpus if "H100" in g.name.upper()]
            use_gpu = h100_gpus[0] if h100_gpus else (gpus[0] if gpus else None)
            logger.info(f"Loading model: {model_name} on GPU: {use_gpu.name if use_gpu else 'CPU'}")
            # (If using CUDA_VISIBLE_DEVICES, set here)
            await self.ollama.pull(model_name)
            self.active_models.add(model_name)
            self.model_gpu_assignments[model_name] = use_gpu.id if use_gpu else None
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def get_model_info(self, model_name: str) -> Optional[ModelProfile]:
        return self.model_profiles.get(model_name)

    async def unload_model(self, model_name: str):
        if model_name in self.active_models:
            self.active_models.remove(model_name)
            logger.info(f"Model {model_name} marked for unloading")

    def get_resource_usage(self) -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        gpus = GPUtil.getGPUs()
        gpu_info = [{
            "id": g.id,
            "name": g.name,
            "memory_used": g.memoryUsed,
            "memory_total": g.memoryTotal,
            "memory_percent": g.memoryUtil * 100,
            "temperature": getattr(g, "temperature", None),
            "load": g.load * 100
        } for g in gpus]
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024 ** 3),
            "gpus": gpu_info,
            "active_models": list(self.active_models),
            "model_gpu_assignments": self.model_gpu_assignments
        }

    def list_available_models(self) -> List[str]:
        return list(self.model_profiles.keys())

    def get_best_model_for_capability(self, capability: str) -> Optional[str]:
        candidates = [k for k, v in self.model_profiles.items() if capability in v.capabilities]
        if not candidates:
            return None
        # Prefer largest, fastest
        return sorted(candidates, key=lambda x: (-self.model_profiles[x].size_gb, self.model_profiles[x].speed))[0]

class MemorySystem:
    """Advanced memory system with vector search, knowledge graph, and H100 optimizations."""

    def __init__(self, persist_directory: str = "./lexos_memory"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self._memory_lock = asyncio.Lock()
        # Use a large, powerful embedding model for H100 (if available)
        try:
            self.embedder = SentenceTransformer('all-mpnet-base-v2')
            logger.info("Using all-mpnet-base-v2 for embeddings.")
        except Exception:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.warning("Falling back to all-MiniLM-L6-v2 embeddings.")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory / "chroma")
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="lexos_memories",
            metadata={"hnsw:space": "cosine"}
        )
        self.memory_index = {}
        self.conversation_history = []
        self.working_memory = []
        self.episodic_memory = []
        self.semantic_memory = {}
        self.memory_graph = nx.Graph()
        self._content_hash_set = set()
        self._load_memories()

    def _load_memories(self):
        memory_file = self.persist_directory / "memory_index.json"
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                self.memory_index = json.load(f)
            for v in self.memory_index.values():
                h = hashlib.sha256(v["content"].encode()).hexdigest()
                self._content_hash_set.add(h)
            logger.info(f"Loaded {len(self.memory_index)} memories")

    def _save_memories(self):
        memory_file = self.persist_directory / "memory_index.json"
        with open(memory_file, 'w') as f:
            json.dump(self.memory_index, f, indent=2)

    async def store_memory(self, content: str, metadata: Dict[str, Any]) -> str:
        async with self._memory_lock:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            if content_hash in self._content_hash_set:
                logger.info("Duplicate memory detected, skipping storage.")
                return ""
            memory_id = hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
            embedding = self.embedder.encode(content).tolist()
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )
            self.memory_index[memory_id] = {
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            self._content_hash_set.add(content_hash)
            self.working_memory.append({
                "id": memory_id,
                "content": content,
                "metadata": metadata
            })
            self.memory_graph.add_node(memory_id)
            await self._update_memory_connections(memory_id, embedding)
            self._save_memories()
            return memory_id

    async def _update_memory_connections(self, memory_id: str, embedding: List[float]):
        # Efficiently connect to top-3 closest memories, building a knowledge graph
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=3
            )
            for i, related_id in enumerate(results['ids'][0]):
                if related_id != memory_id and results['distances'][0][i] < 0.4:
                    self.memory_graph.add_edge(memory_id, related_id, weight=1-results['distances'][0][i])
        except Exception as e:
            logger.warning(f"Could not update memory graph for {memory_id}: {e}")

    async def recall_memories(self, query: str, n_results: int = 5, include_graph: bool = True) -> List[Dict]:
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        memories = []
        for i in range(len(results['ids'][0])):
            mem = {
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            }
            memories.append(mem)
        # Optionally, augment with graph-neighbor memories for rich context
        if include_graph and memories:
            graph_ids = set()
            for mem in memories:
                graph_ids.update(list(self.memory_graph.neighbors(mem['id'])) if self.memory_graph.has_node(mem['id']) else [])
            for gid in graph_ids:
                if gid not in [m["id"] for m in memories]:
                    doc = self.memory_index.get(gid)
                    if doc:
                        memories.append({
                            "id": gid,
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "distance": 0.5  # Graph-based, less relevant
                        })
        return sorted(memories, key=lambda x: x["distance"])[:n_results]

    async def consolidate_memories(self):
        """Consolidate short-term memories into long-term storage, auto-prune old memories."""
        async with self._memory_lock:
            if len(self.working_memory) > 10:
                memories_to_consolidate = self.working_memory[:-5]
                self.episodic_memory.extend(memories_to_consolidate)
                self.working_memory = self.working_memory[-5:]
                # Extract key facts for semantic memory
                for memory in memories_to_consolidate:
                    if "task_type" in memory.get("metadata", {}):
                        task_type = memory["metadata"]["task_type"]
                        if task_type not in self.semantic_memory:
                            self.semantic_memory[task_type] = []
                        self.semantic_memory[task_type].append({
                            "content": memory["content"][:200],
                            "timestamp": memory["metadata"].get("timestamp", "")
                        })
                logger.info(f"Consolidated {len(memories_to_consolidate)} memories")
            # Prune memories older than 1M if memory pressure
            if len(self.memory_index) > 5000:
                prune_ids = sorted(self.memory_index, key=lambda k: self.memory_index[k]['timestamp'])[:100]
                for pid in prune_ids:
                    self.delete_memory(pid)
                logger.info(f"Pruned {len(prune_ids)} oldest memories")

    def add_conversation(self, user_input: str, ai_response: str):
        self.conversation_history.append({
            "user": user_input,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def get_conversation_context(self, n_turns: int = 5) -> str:
        recent = self.conversation_history[-n_turns:]
        context = []
        for turn in recent:
            context.append(f"User: {turn['user']}")
            context.append(f"Assistant: {turn['assistant']}")
        return "\n".join(context)

    def search_semantic_memory(self, category: str) -> List[Dict]:
        return self.semantic_memory.get(category, [])

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": len(self.memory_index),
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "conversation_history_size": len(self.conversation_history),
            "semantic_categories": list(self.semantic_memory.keys()),
            "vector_store_size": self.collection.count(),
            "graph_nodes": self.memory_graph.number_of_nodes(),
            "graph_edges": self.memory_graph.number_of_edges()
        }

    def search_memories_by_keyword(self, keyword: str, n: int = 5) -> List[Dict]:
        matches = []
        for mid, mem in self.memory_index.items():
            if keyword.lower() in mem["content"].lower():
                matches.append({"id": mid, **mem})
            if len(matches) >= n:
                break
        return matches

    def delete_memory(self, memory_id: str) -> bool:
        removed = False
        if memory_id in self.memory_index:
            content = self.memory_index[memory_id]["content"]
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            self._content_hash_set.discard(content_hash)
            del self.memory_index[memory_id]
            removed = True
        self.working_memory = [m for m in self.working_memory if m.get("id") != memory_id]
        self.episodic_memory = [m for m in self.episodic_memory if m.get("id") != memory_id]
        try:
            self.collection.delete(ids=[memory_id])
        except Exception as e:
            logger.warning(f"Failed to delete memory from vector db: {e}")
        if self.memory_graph.has_node(memory_id):
            self.memory_graph.remove_node(memory_id)
        self._save_memories()
        return removed

    async def batch_store_memories(self, contents: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Batch memory storage - optimized for H100 batch throughput."""
        async with self._memory_lock:
            memory_ids = []
            embeddings = self.embedder.encode(contents)
            for idx, content in enumerate(contents):
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                if content_hash in self._content_hash_set:
                    continue
                memory_id = hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
                self.collection.add(
                    ids=[memory_id],
                    embeddings=[embeddings[idx].tolist()],
                    documents=[content],
                    metadatas=[metadatas[idx]]
                )
                self.memory_index[memory_id] = {
                    "content": content,
                    "metadata": metadatas[idx],
                    "timestamp": datetime.now().isoformat()
                }
                self._content_hash_set.add(content_hash)
                self.working_memory.append({
                    "id": memory_id,
                    "content": content,
                    "metadata": metadatas[idx]
                })
                self.memory_graph.add_node(memory_id)
                await self._update_memory_connections(memory_id, embeddings[idx].tolist())
                memory_ids.append(memory_id)
            self._save_memories()
            return memory_ids

    def memory_gc(self):
        gc.collect()
        logger.info("Manual garbage collection triggered.")
