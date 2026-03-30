"""Layer-wise model loading for low-memory inference.

This module implements AirLLM-style layer-wise model loading, where:
- Model weights are split into individual layer shards
- Layers are loaded on-demand during inference
- Only the active layer + prefetch buffer is in GPU memory
- CPU memory can be used as a secondary cache

Key features:
- Model split into layer shards (each layer is a separate file)
- On-demand layer loading during inference
- Prefetching overlaps loading and compute
- VRAM usage <4GB for 70B model
- Works with both CPU and GPU

References:
    - AirLLM: Compress your LLM using layer-wise short GPTQ and layer-wise streaming
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file, save_file

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Memory thresholds
DEFAULT_LAYER_CACHE_SIZE = 2  # Keep 2 layers in GPU (current + prefetch)
DEFAULT_PREFETCH_AHEAD = 1  # Prefetch 1 layer ahead
MAX_VRAM_GB = 4.0  # Target maximum VRAM usage in GB


class LayerShardMetadata:
    """Metadata for a layer shard.

    Attributes:
        layer_idx: Layer index in the model
        layer_type: Type of layer (e.g., "transformer", "attention", "mlp")
        param_count: Number of parameters in this layer
        param_names: List of parameter names in this layer
        file_path: Path to the shard file
        device: Device where this layer is currently loaded (if any)
        last_accessed: Timestamp of last access for LRU eviction
    """

    def __init__(
        self,
        layer_idx: int,
        layer_type: str,
        param_names: list[str],
        file_path: str | Path,
    ) -> None:
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.param_names = param_names
        self.param_count = len(param_names)
        self.file_path = Path(file_path)
        self.device: str | None = None
        self.last_accessed: float = 0.0
        self.size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_idx": self.layer_idx,
            "layer_type": self.layer_type,
            "param_names": self.param_names,
            "param_count": self.param_count,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], file_path: str | Path) -> LayerShardMetadata:
        """Create from dictionary."""
        meta = cls(
            layer_idx=data["layer_idx"],
            layer_type=data["layer_type"],
            param_names=data["param_names"],
            file_path=file_path,
        )
        meta.size_bytes = data.get("size_bytes", 0)
        return meta


class ModelShardIndex:
    """Index of all layer shards for a model.

    This index tracks which layers exist, their metadata,
    and provides utilities for layer management.
    """

    def __init__(self, model_id: str, shard_dir: str | Path) -> None:
        self.model_id = model_id
        self.shard_dir = Path(shard_dir)
        self.layers: dict[int, LayerShardMetadata] = {}
        self.embedding_layer: LayerShardMetadata | None = None
        self.lm_head: LayerShardMetadata | None = None
        self.norm_layer: LayerShardMetadata | None = None
        self.total_params: int = 0
        self._index_file = self.shard_dir / "shard_index.json"

    def add_layer(self, metadata: LayerShardMetadata) -> None:
        """Add a layer to the index."""
        self.layers[metadata.layer_idx] = metadata
        self.total_params += metadata.param_count

    def get_layer(self, layer_idx: int) -> LayerShardMetadata | None:
        """Get metadata for a specific layer."""
        return self.layers.get(layer_idx)

    def get_all_layer_indices(self) -> list[int]:
        """Get all layer indices in sorted order."""
        return sorted(self.layers.keys())

    def save(self) -> None:
        """Save the index to disk."""
        data = {
            "model_id": self.model_id,
            "total_layers": len(self.layers),
            "total_params": self.total_params,
            "layers": {idx: meta.to_dict() for idx, meta in self.layers.items()},
        }

        if self.embedding_layer:
            data["embedding_layer"] = self.embedding_layer.to_dict()
        if self.lm_head:
            data["lm_head"] = self.lm_head.to_dict()
        if self.norm_layer:
            data["norm_layer"] = self.norm_layer.to_dict()

        self._index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved shard index to {self._index_file}")

    def load(self) -> bool:
        """Load the index from disk.

        Returns:
            True if index was loaded successfully, False otherwise
        """
        if not self._index_file.exists():
            return False

        try:
            with open(self._index_file) as f:
                data = json.load(f)

            self.model_id = data.get("model_id", self.model_id)
            self.total_params = data.get("total_params", 0)

            # Load layers
            for idx, layer_data in data.get("layers", {}).items():
                file_path = self.shard_dir / f"layer_{idx}.safetensors"
                meta = LayerShardMetadata.from_dict(layer_data, file_path)
                self.layers[int(idx)] = meta

            # Load special layers
            if "embedding_layer" in data:
                self.embedding_layer = LayerShardMetadata.from_dict(
                    data["embedding_layer"],
                    self.shard_dir / "embedding.safetensors"
                )
            if "lm_head" in data:
                self.lm_head = LayerShardMetadata.from_dict(
                    data["lm_head"],
                    self.shard_dir / "lm_head.safetensors"
                )
            if "norm_layer" in data:
                self.norm_layer = LayerShardMetadata.from_dict(
                    data["norm_layer"],
                    self.shard_dir / "norm.safetensors"
                )

            logger.info(f"Loaded shard index: {len(self.layers)} layers")
            return True

        except Exception as e:
            logger.error(f"Failed to load shard index: {e}")
            return False

    @classmethod
    def from_sharded_model(cls, model_id: str, shard_dir: str | Path) -> ModelShardIndex:
        """Load index from an existing sharded model directory."""
        index = cls(model_id, shard_dir)
        index.load()
        return index


class LayerShardManager:
    """Manages splitting a model into layer shards.

    This class handles:
    - Analyzing model structure to identify layers
    - Splitting weights into individual layer files
    - Creating and managing the shard index
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str | Path,
        quantization_bits: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.quantization_bits = quantization_bits
        self.shard_dir = self.output_dir / "shards"
        self.index = ModelShardIndex(model_id, self.shard_dir)

    def shard_model(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_pattern: str | None = None,
    ) -> Path:
        """Split a model state dict into layer shards.

        Args:
            state_dict: Full model state dictionary
            layer_pattern: Regex pattern to identify layer names
                (default: matches "model.layers.N" or "transformer.h.N")

        Returns:
            Path to the shard directory
        """
        import re

        logger.info(f"Sharding model {self.model_id}...")

        # Default patterns for common architectures
        if layer_pattern is None:
            # Try different patterns
            for pattern in [
                r"model\.layers\.(\d+)",
                r"transformer\.h\.(\d+)",
                r"transformer\.layer\.(\d+)",
                r"encoder\.layer\.(\d+)",
                r"decoder\.layers\.(\d+)",
                r"layers\.(\d+)",
            ]:
                if any(re.search(pattern, name) for name in state_dict):
                    layer_pattern = pattern
                    logger.info(f"Detected layer pattern: {pattern}")
                    break

        if layer_pattern is None:
            # Fallback: group by common prefix
            layer_pattern = r"^(.*?\.(\d+)\.[^.]+)"
            logger.info("Using fallback layer pattern")

        # Group parameters by layer
        layer_groups: dict[int, dict[str, torch.Tensor]] = {}
        other_params: dict[str, torch.Tensor] = {}

        pattern_re = re.compile(layer_pattern)

        for name, tensor in state_dict.items():
            match = pattern_re.search(name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = {}
                layer_groups[layer_idx][name] = tensor
            else:
                other_params[name] = tensor

        # Create shard directory
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        # Save each layer as a separate shard
        total_size = 0
        for layer_idx, layer_params in sorted(layer_groups.items()):
            shard_file = self.shard_dir / f"layer_{layer_idx}.safetensors"
            save_file(layer_params, str(shard_file))

            file_size = shard_file.stat().st_size
            total_size += file_size

            meta = LayerShardMetadata(
                layer_idx=layer_idx,
                layer_type="transformer",
                param_names=list(layer_params.keys()),
                file_path=shard_file,
            )
            meta.size_bytes = file_size
            self.index.add_layer(meta)

            logger.debug(f"Shard layer {layer_idx}: {len(layer_params)} params, {file_size / 1e6:.1f} MB")

        # Handle embedding and output layers separately
        self._save_special_layers(other_params)

        # Save index
        self.index.save()

        logger.info(
            f"Sharding complete: {len(layer_groups)} layers, "
            f"{total_size / 1e9:.2f} GB total"
        )

        return self.shard_dir

    def _save_special_layers(self, params: dict[str, torch.Tensor]) -> None:
        """Save embedding, norm, and output layers separately."""
        # Group special layers
        embed_params = {k: v for k, v in params.items() if "embed" in k.lower()}
        norm_params = {k: v for k, v in params.items() if "norm" in k.lower()}
        lm_head_params = {k: v for k, v in params.items() if any(x in k.lower() for x in ["lm_head", "output", "head"])}
        other_remaining = {k: v for k, v in params.items() if k not in set(embed_params) | set(norm_params) | set(lm_head_params)}

        if embed_params:
            shard_file = self.shard_dir / "embedding.safetensors"
            save_file(embed_params, str(shard_file))
            self.index.embedding_layer = LayerShardMetadata(
                layer_idx=-1,
                layer_type="embedding",
                param_names=list(embed_params.keys()),
                file_path=shard_file,
            )
            self.index.embedding_layer.size_bytes = shard_file.stat().st_size
            logger.info(f"Saved embedding layer: {len(embed_params)} params")

        if norm_params:
            shard_file = self.shard_dir / "norm.safetensors"
            save_file(norm_params, str(shard_file))
            self.index.norm_layer = LayerShardMetadata(
                layer_idx=-2,
                layer_type="norm",
                param_names=list(norm_params.keys()),
                file_path=shard_file,
            )
            self.index.norm_layer.size_bytes = shard_file.stat().st_size
            logger.info(f"Saved norm layer: {len(norm_params)} params")

        if lm_head_params:
            shard_file = self.shard_dir / "lm_head.safetensors"
            save_file(lm_head_params, str(shard_file))
            self.index.lm_head = LayerShardMetadata(
                layer_idx=-3,
                layer_type="lm_head",
                param_names=list(lm_head_params.keys()),
                file_path=shard_file,
            )
            self.index.lm_head.size_bytes = shard_file.stat().st_size
            logger.info(f"Saved LM head: {len(lm_head_params)} params")

        if other_remaining:
            shard_file = self.shard_dir / "other.safetensors"
            save_file(other_remaining, str(shard_file))
            logger.info(f"Saved {len(other_remaining)} other params")


class LayerCache:
    """LRU cache for loaded layer tensors.

    Manages which layers are currently in memory (CPU or GPU),
    with automatic eviction when capacity is exceeded.
    """

    def __init__(
        self,
        max_gpu_layers: int = DEFAULT_LAYER_CACHE_SIZE,
        max_cpu_layers: int = 4,
        target_device: str = "cuda",
    ) -> None:
        self.max_gpu_layers = max_gpu_layers
        self.max_cpu_layers = max_cpu_layers
        self.target_device = target_device if torch.cuda.is_available() else "cpu"
        self.cpu_device = "cpu"

        # Cache storage: layer_idx -> dict of tensors
        self.gpu_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.cpu_cache: dict[int, dict[str, torch.Tensor]] = {}

        # Access tracking for LRU
        self.gpu_access_time: dict[int, float] = {}
        self.cpu_access_time: dict[int, float] = {}

        self._lock = threading.Lock()
        self._access_counter = 0

    def get(self, layer_idx: int) -> dict[str, torch.Tensor] | None:
        """Get a layer from cache if available.

        Returns the layer tensors, preferring GPU if available.
        Also updates access time for LRU tracking.
        """
        with self._lock:
            self._access_counter += 1

            # Check GPU first
            if layer_idx in self.gpu_cache:
                self.gpu_access_time[layer_idx] = self._access_counter
                return self.gpu_cache[layer_idx]

            # Check CPU
            if layer_idx in self.cpu_cache:
                self.cpu_access_time[layer_idx] = self._access_counter
                return self.cpu_cache[layer_idx]

            return None

    def put_gpu(self, layer_idx: int, tensors: dict[str, torch.Tensor]) -> None:
        """Store layer in GPU cache, evicting if necessary."""
        with self._lock:
            # Evict oldest GPU layer if at capacity
            while len(self.gpu_cache) >= self.max_gpu_layers:
                self._evict_oldest_gpu()

            self.gpu_cache[layer_idx] = tensors
            self.gpu_access_time[layer_idx] = self._access_counter

    def put_cpu(self, layer_idx: int, tensors: dict[str, torch.Tensor]) -> None:
        """Store layer in CPU cache, evicting if necessary."""
        with self._lock:
            # Evict oldest CPU layer if at capacity
            while len(self.cpu_cache) >= self.max_cpu_layers:
                self._evict_oldest_cpu()

            self.cpu_cache[layer_idx] = tensors
            self.cpu_access_time[layer_idx] = self._access_counter

    def _evict_oldest_gpu(self) -> None:
        """Evict least recently used layer from GPU."""
        if not self.gpu_cache:
            return

        # Find oldest
        oldest_idx = min(self.gpu_cache.keys(), key=lambda k: self.gpu_access_time.get(k, 0))

        # Move to CPU if space available, otherwise delete
        if len(self.cpu_cache) < self.max_cpu_layers:
            tensors = self.gpu_cache[oldest_idx]
            # Move to CPU
            cpu_tensors = {k: v.cpu() for k, v in tensors.items()}
            self.cpu_cache[oldest_idx] = cpu_tensors
            self.cpu_access_time[oldest_idx] = self.gpu_access_time[oldest_idx]

        # Clear from GPU
        del self.gpu_cache[oldest_idx]
        del self.gpu_access_time[oldest_idx]

        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evict_oldest_cpu(self) -> None:
        """Evict least recently used layer from CPU."""
        if not self.cpu_cache:
            return

        oldest_idx = min(self.cpu_cache.keys(), key=lambda k: self.cpu_access_time.get(k, 0))
        del self.cpu_cache[oldest_idx]
        del self.cpu_access_time[oldest_idx]

    def move_to_gpu(self, layer_idx: int) -> dict[str, torch.Tensor] | None:
        """Move a layer from CPU to GPU cache."""
        with self._lock:
            if layer_idx not in self.cpu_cache:
                return None

            # Evict if necessary
            while len(self.gpu_cache) >= self.max_gpu_layers:
                self._evict_oldest_gpu()

            # Move to GPU
            tensors = self.cpu_cache[layer_idx]
            gpu_tensors = {k: v.to(self.target_device) for k, v in tensors.items()}

            self.gpu_cache[layer_idx] = gpu_tensors
            self.gpu_access_time[layer_idx] = self._access_counter

            # Remove from CPU cache
            del self.cpu_cache[layer_idx]
            del self.cpu_access_time[layer_idx]

            return gpu_tensors

    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self.gpu_cache.clear()
            self.cpu_cache.clear()
            self.gpu_access_time.clear()
            self.cpu_access_time.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {
            "gpu_layers_cached": len(self.gpu_cache),
            "cpu_layers_cached": len(self.cpu_cache),
            "gpu_max_layers": self.max_gpu_layers,
            "cpu_max_layers": self.max_cpu_layers,
        }

        if torch.cuda.is_available():
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        return stats


class LayerPrefetcher:
    """Prefetches layers during inference to overlap loading and compute.

    Uses a background thread to load upcoming layers while the current
    layer is being computed.
    """

    def __init__(
        self,
        shard_index: ModelShardIndex,
        layer_cache: LayerCache,
        prefetch_ahead: int = DEFAULT_PREFETCH_AHEAD,
    ) -> None:
        self.shard_index = shard_index
        self.layer_cache = layer_cache
        self.prefetch_ahead = prefetch_ahead

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        self._prefetch_futures: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    def start_prefetch(self, current_layer_idx: int) -> None:
        """Start prefetching layers that will be needed next.

        Args:
            current_layer_idx: The layer currently being executed
        """
        if self._shutdown:
            return

        all_indices = self.shard_index.get_all_layer_indices()

        # Find position in sequence
        try:
            current_pos = all_indices.index(current_layer_idx)
        except ValueError:
            return

        # Prefetch next N layers
        for i in range(1, self.prefetch_ahead + 1):
            next_pos = current_pos + i
            if next_pos >= len(all_indices):
                break

            next_idx = all_indices[next_pos]

            # Skip if already in cache
            if self.layer_cache.get(next_idx) is not None:
                continue

            # Skip if already prefetching
            with self._lock:
                if next_idx in self._prefetch_futures:
                    continue

            # Start prefetch
            future = self._executor.submit(self._load_layer, next_idx)
            with self._lock:
                self._prefetch_futures[next_idx] = future

    def _load_layer(self, layer_idx: int) -> None:
        """Load a layer into CPU cache (called in background thread)."""
        try:
            meta = self.shard_index.get_layer(layer_idx)
            if meta is None:
                return

            # Check again if already loaded
            if self.layer_cache.get(layer_idx) is not None:
                return

            # Load from disk to CPU
            tensors = load_file(str(meta.file_path), device="cpu")

            # Store in CPU cache
            self.layer_cache.put_cpu(layer_idx, tensors)

            logger.debug(f"Prefetched layer {layer_idx} to CPU cache")

        except Exception as e:
            logger.warning(f"Failed to prefetch layer {layer_idx}: {e}")

        finally:
            with self._lock:
                self._prefetch_futures.pop(layer_idx, None)

    def get_prefetched_layer(self, layer_idx: int) -> dict[str, torch.Tensor] | None:
        """Get a prefetched layer if available, waiting if necessary."""
        with self._lock:
            future = self._prefetch_futures.get(layer_idx)

        if future:
            try:
                future.result(timeout=30)  # Wait up to 30 seconds
            except Exception as e:
                logger.warning(f"Prefetch wait failed for layer {layer_idx}: {e}")

        # Now check the cache
        return self.layer_cache.get(layer_idx)

    def shutdown(self) -> None:
        """Shutdown the prefetcher."""
        self._shutdown = True

        # Cancel pending futures
        with self._lock:
            for future in self._prefetch_futures.values():
                future.cancel()
            self._prefetch_futures.clear()

        self._executor.shutdown(wait=False)


class LayerWiseLoader:
    """Main class for layer-wise model loading during inference.

    This class manages:
    - Loading layers on-demand from disk
    - Caching recently used layers
    - Prefetching upcoming layers
    - Tracking memory usage

    Example:
        >>> loader = LayerWiseLoader(
        ...     shard_dir="/path/to/shards",
        ...     max_gpu_layers=2,
        ... )
        >>> # During forward pass
        >>> for layer_idx in range(num_layers):
        ...     layer_weights = loader.load_layer(layer_idx)
        ...     # Execute layer computation
        ...     output = transformer_layer(output, layer_weights)
        ...     # Release layer after use
        ...     loader.release_layer(layer_idx)
    """

    def __init__(
        self,
        shard_dir: str | Path,
        max_gpu_layers: int = DEFAULT_LAYER_CACHE_SIZE,
        max_cpu_layers: int = 4,
        prefetch_ahead: int = DEFAULT_PREFETCH_AHEAD,
        device: str = "cuda",
    ) -> None:
        """Initialize the layer-wise loader.

        Args:
            shard_dir: Directory containing the sharded model
            max_gpu_layers: Maximum layers to keep in GPU memory
            max_cpu_layers: Maximum layers to keep in CPU memory
            prefetch_ahead: Number of layers to prefetch ahead
            device: Target device ("cuda" or "cpu")
        """
        self.shard_dir = Path(shard_dir)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_gpu_layers = max_gpu_layers
        self.max_cpu_layers = max_cpu_layers

        # Load shard index
        self.index = ModelShardIndex.from_sharded_model(
            model_id="unknown",  # Will be read from index
            shard_dir=self.shard_dir,
        )

        # Initialize cache
        self.cache = LayerCache(
            max_gpu_layers=max_gpu_layers,
            max_cpu_layers=max_cpu_layers,
            target_device=self.device,
        )

        # Initialize prefetcher
        self.prefetcher = LayerPrefetcher(
            shard_index=self.index,
            layer_cache=self.cache,
            prefetch_ahead=prefetch_ahead,
        )

        # Load special layers immediately
        self.special_layers: dict[str, dict[str, torch.Tensor]] = {}
        self._load_special_layers()

        self._current_layer: int | None = None
        self._layer_load_times: list[float] = []

    def _load_special_layers(self) -> None:
        """Load embedding, norm, and LM head layers."""
        if self.index.embedding_layer:
            tensors = load_file(str(self.index.embedding_layer.file_path), device=self.device)
            self.special_layers["embedding"] = tensors
            logger.info("Loaded embedding layer")

        if self.index.norm_layer:
            tensors = load_file(str(self.index.norm_layer.file_path), device=self.device)
            self.special_layers["norm"] = tensors
            logger.info("Loaded norm layer")

        if self.index.lm_head:
            tensors = load_file(str(self.index.lm_head.file_path), device=self.device)
            self.special_layers["lm_head"] = tensors
            logger.info("Loaded LM head")

    def load_layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Load a layer's weights for computation.

        This method:
        1. Checks if the layer is already in cache
        2. If in CPU cache, moves to GPU
        3. If not cached, loads from disk
        4. Triggers prefetch for upcoming layers

        Args:
            layer_idx: Index of the layer to load

        Returns:
            Dictionary of layer tensors on the target device

        Raises:
            KeyError: If layer_idx is not found in the index
        """
        start_time = time.time()
        self._current_layer = layer_idx

        # Check cache first
        cached = self.cache.get(layer_idx)
        if cached is not None:
            # If in CPU cache, move to GPU
            if list(cached.values())[0].device.type != self.device:
                cached = self.cache.move_to_gpu(layer_idx)
                if cached is None:
                    raise RuntimeError(f"Failed to move layer {layer_idx} to GPU")

            self._record_load_time(time.time() - start_time, cached=True)
            self.prefetcher.start_prefetch(layer_idx)
            return cached

        # Check if prefetched
        prefetched = self.prefetcher.get_prefetched_layer(layer_idx)
        if prefetched is not None:
            # Move to GPU
            gpu_tensors = self.cache.move_to_gpu(layer_idx)
            if gpu_tensors is None:
                # Load directly to GPU
                meta = self.index.get_layer(layer_idx)
                if meta is None:
                    raise KeyError(f"Layer {layer_idx} not found in index")

                tensors = load_file(str(meta.file_path), device=self.device)
                self.cache.put_gpu(layer_idx, tensors)
                gpu_tensors = tensors

            self._record_load_time(time.time() - start_time, prefetched=True)
            self.prefetcher.start_prefetch(layer_idx)
            return gpu_tensors

        # Load from disk
        meta = self.index.get_layer(layer_idx)
        if meta is None:
            raise KeyError(f"Layer {layer_idx} not found in index")

        tensors = load_file(str(meta.file_path), device=self.device)
        self.cache.put_gpu(layer_idx, tensors)

        self._record_load_time(time.time() - start_time, cached=False)
        self.prefetcher.start_prefetch(layer_idx)

        return tensors

    def _record_load_time(self, duration: float, cached: bool = False, prefetched: bool = False) -> None:
        """Record layer load time for metrics."""
        self._layer_load_times.append(duration)

        load_type = "cache" if cached else ("prefetch" if prefetched else "disk")
        logger.debug(f"Layer {self._current_layer} loaded from {load_type}: {duration*1000:.1f}ms")

    def release_layer(self, layer_idx: int) -> None:
        """Release a layer after computation (hint for cache management).

        Currently this is a no-op as the cache handles eviction automatically,
        but it can be used in the future for more aggressive eviction.

        Args:
            layer_idx: Index of the layer to release
        """
        pass  # Cache handles eviction automatically

    def get_special_layer(self, layer_type: str) -> dict[str, torch.Tensor] | None:
        """Get a special layer (embedding, norm, lm_head).

        Args:
            layer_type: One of "embedding", "norm", "lm_head"

        Returns:
            Layer tensors or None if not available
        """
        return self.special_layers.get(layer_type)

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory usage statistics."""
        stats = self.cache.get_memory_stats()
        stats["avg_load_time_ms"] = sum(self._layer_load_times) / len(self._layer_load_times) * 1000 if self._layer_load_times else 0
        stats["total_layers"] = len(self.index.layers)
        return stats

    def estimate_vram_usage(self, model_size_gb: float) -> float:
        """Estimate VRAM usage for a model.

        Args:
            model_size_gb: Total model size in GB

        Returns:
            Estimated VRAM usage in GB
        """
        if self.device == "cpu":
            return 0.0

        # Each layer: (model_size / num_layers) * max_gpu_layers
        num_layers = len(self.index.layers)
        if num_layers == 0:
            return model_size_gb

        layer_size_gb = model_size_gb / num_layers
        estimated_gb = layer_size_gb * self.max_gpu_layers

        # Add overhead for special layers and activations
        overhead_gb = 0.5

        return estimated_gb + overhead_gb

    def shutdown(self) -> None:
        """Shutdown the loader and free all resources."""
        self.prefetcher.shutdown()
        self.cache.clear()
        self.special_layers.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Layer-wise loader shutdown complete")


class LayerWiseInferenceEngine:
    """High-level inference engine using layer-wise loading.

    This engine orchestrates the entire inference process:
    - Token embedding
    - Layer-by-layer forward pass with on-demand loading
    - Final output generation

    Example:
        >>> engine = LayerWiseInferenceEngine(
        ...     shard_dir="/path/to/shards",
        ...     max_gpu_layers=2,
        ... )
        >>> # Run inference
        >>> output = engine.generate(input_ids, max_tokens=50)
    """

    def __init__(
        self,
        shard_dir: str | Path,
        max_gpu_layers: int = DEFAULT_LAYER_CACHE_SIZE,
        max_cpu_layers: int = 4,
        prefetch_ahead: int = DEFAULT_PREFETCH_AHEAD,
        device: str = "cuda",
    ) -> None:
        self.loader = LayerWiseLoader(
            shard_dir=shard_dir,
            max_gpu_layers=max_gpu_layers,
            max_cpu_layers=max_cpu_layers,
            prefetch_ahead=prefetch_ahead,
            device=device,
        )
        self.device = device if torch.cuda.is_available() else "cpu"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_executor: Callable | None = None,
    ) -> torch.Tensor:
        """Run forward pass through all layers.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            layer_executor: Callable that executes a single layer forward pass
                If None, returns embeddings for testing

        Returns:
            Output logits or hidden states
        """
        # Get embeddings
        embed_layer = self.loader.get_special_layer("embedding")
        if embed_layer is None:
            raise RuntimeError("No embedding layer found")

        # Simple embedding lookup
        hidden_states = self._embed_tokens(input_ids, embed_layer)
        hidden_states = hidden_states.to(self.device)

        if layer_executor is None:
            # Return embeddings if no executor provided (testing)
            return hidden_states

        # Run through transformer layers
        layer_indices = self.loader.index.get_all_layer_indices()

        for layer_idx in layer_indices:
            # Load layer weights
            layer_weights = self.loader.load_layer(layer_idx)

            # Execute layer
            hidden_states = layer_executor(
                hidden_states,
                layer_weights,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
            )

        # Apply norm and LM head
        norm_layer = self.loader.get_special_layer("norm")
        lm_head = self.loader.get_special_layer("lm_head")

        if norm_layer:
            hidden_states = self._apply_norm(hidden_states, norm_layer)

        if lm_head:
            logits = self._apply_lm_head(hidden_states, lm_head)
            return logits

        return hidden_states

    def _embed_tokens(self, input_ids: torch.Tensor, embed_weights: dict[str, torch.Tensor]) -> torch.Tensor:
        """Simple token embedding lookup."""
        # Look for weight tensor
        weight_key = next((k for k in embed_weights if "weight" in k.lower()), None)
        if weight_key is None:
            weight_key = list(embed_weights)[0]

        weight = embed_weights[weight_key]
        return torch.nn.functional.embedding(input_ids, weight)

    def _apply_norm(self, hidden_states: torch.Tensor, norm_weights: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply layer normalization."""
        # This is a simplified version - real implementation would need proper LayerNorm
        return hidden_states

    def _apply_lm_head(self, hidden_states: torch.Tensor, lm_head_weights: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply LM head to get logits."""
        # Simple linear projection
        weight_key = next((k for k in lm_head_weights if "weight" in k.lower()), None)
        if weight_key is None:
            weight_key = list(lm_head_weights)[0]

        weight = lm_head_weights[weight_key]
        return torch.matmul(hidden_states, weight.t())

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory usage statistics."""
        return self.loader.get_memory_stats()

    def shutdown(self) -> None:
        """Shutdown the engine."""
        self.loader.shutdown()


# Convenience functions for CLI integration

def shard_model_for_layerwise_loading(
    model_id: str,
    state_dict: dict[str, torch.Tensor],
    output_dir: str | Path,
    quantization_bits: int | None = None,
) -> Path:
    """Split a model into layer shards for layer-wise loading.

    This is a convenience function for the CLI quantize command.

    Args:
        model_id: Model identifier
        state_dict: Model state dictionary
        output_dir: Directory to save shards
        quantization_bits: Optional quantization bit width

    Returns:
        Path to the shard directory
    """
    manager = LayerShardManager(
        model_id=model_id,
        output_dir=output_dir,
        quantization_bits=quantization_bits,
    )
    return manager.shard_model(state_dict)


def create_layerwise_loader(
    shard_dir: str | Path,
    max_gpu_layers: int = DEFAULT_LAYER_CACHE_SIZE,
    device: str = "cuda",
) -> LayerWiseLoader:
    """Create a layer-wise loader for inference.

    Args:
        shard_dir: Directory containing model shards
        max_gpu_layers: Maximum layers to keep in GPU
        device: Target device

    Returns:
        Configured LayerWiseLoader instance
    """
    return LayerWiseLoader(
        shard_dir=shard_dir,
        max_gpu_layers=max_gpu_layers,
        device=device,
    )


__all__ = [
    "LayerShardMetadata",
    "ModelShardIndex",
    "LayerShardManager",
    "LayerCache",
    "LayerPrefetcher",
    "LayerWiseLoader",
    "LayerWiseInferenceEngine",
    "shard_model_for_layerwise_loading",
    "create_layerwise_loader",
    "DEFAULT_LAYER_CACHE_SIZE",
    "DEFAULT_PREFETCH_AHEAD",
    "MAX_VRAM_GB",
]
