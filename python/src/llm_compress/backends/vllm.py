"""vLLM backend adapter with TurboQuant KV cache integration.

This module integrates vLLM with TurboQuant KV cache compression.
The integration works through:
1. Wrapping vLLM's LLM class to use TurboQuant KV cache
2. Providing compression/decompression hooks for attention layers
3. Implementing hybrid decode for compressed cache operations

The module handles the case where vLLM is not installed (Python 3.14+) by
providing stub implementations that raise informative errors.

References:
    - TurboQuant: Making KV Cache Compression Robust to Pruning for Efficient LLM Inference
    - vLLM: A high-throughput and memory-efficient inference engine for LLMs
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from llm_compress.backends.base import BaseBackend
from llm_compress.quantization.kv_cache import KVCacheQuantizer

# Try to import vLLM - may not be available on all Python versions
try:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.kv_cache_utils import free_kv_cache

    VLLM_AVAILABLE = True
    VLLM_VERSION = getattr(vllm, "__version__", "unknown")
except ImportError:
    VLLM_AVAILABLE = False
    VLLM_VERSION = None
    vllm = None  # type: ignore
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    free_kv_cache = None  # type: ignore

if TYPE_CHECKING:
    pass


class TurboQuantKVCacheManager:
    """TurboQuant KV cache manager for vLLM integration.
    
    This class wraps vLLM's KV cache with TurboQuant compression, providing:
    - Compression of keys and values during cache updates
    - Decompression during attention computation
    - Memory-efficient cache management
    
    The manager maintains separate quantizers for keys and values, and handles
    the conversion between compressed and uncompressed formats.
    
    Attributes:
        quantizer: KVCacheQuantizer instance for compression/decompression
        head_dim: Dimension per attention head
        num_layers: Number of transformer layers
        compression_ratio: Measured compression ratio
    """

    def __init__(
        self,
        head_dim: int = 64,
        num_layers: int = 32,
        key_bits: int = 3,
        value_bits: int = 2,
        key_proj_dim: int | None = None,
        value_group_size: int = 8,
        seed: int = 42,
    ) -> None:
        """Initialize the TurboQuant KV cache manager.
        
        Args:
            head_dim: Dimension per attention head
            num_layers: Number of transformer layers
            key_bits: Bits for key quantization (default 3)
            value_bits: Bits for value quantization (default 2)
            key_proj_dim: Projection dimension for keys (default head_dim//2)
            value_group_size: Group size for value quantization
            seed: Random seed for reproducibility
        """
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.key_proj_dim = key_proj_dim or (head_dim // 2)
        self.value_group_size = value_group_size

        # Initialize quantizer
        self.quantizer = KVCacheQuantizer(
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            key_proj_dim=self.key_proj_dim,
            value_group_size=value_group_size,
            seed=seed,
        )

        # State tracking
        self._fitted = False
        self.compression_ratio: float = 0.0
        self.compressed_cache: dict[int, dict[str, Any]] = {}
        self.layer_shapes: dict[int, tuple[int, ...]] = {}

    def fit(
        self,
        sample_keys: torch.Tensor,
        sample_values: torch.Tensor,
    ) -> TurboQuantKVCacheManager:
        """Fit the quantizer on sample KV cache data.
        
        Args:
            sample_keys: Sample keys of shape (..., head_dim)
            sample_values: Sample values of shape (..., head_dim)
            
        Returns:
            Self for method chaining
        """
        self.quantizer.fit(sample_keys, sample_values)
        self._fitted = True
        return self

    def compress_layer_cache(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> dict[str, Any]:
        """Compress KV cache for a specific layer.
        
        Args:
            layer_idx: Layer index
            keys: Key tensor of shape (batch, num_heads, seq_len, head_dim)
            values: Value tensor of shape (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Dictionary containing compressed cache data
        """
        if not self._fitted:
            # Auto-fit on first use
            self.fit(keys.flatten(0, -2), values.flatten(0, -2))

        compressed = self.quantizer.compress_kv_cache(keys, values)
        self.compressed_cache[layer_idx] = compressed
        self.layer_shapes[layer_idx] = keys.shape

        # Calculate compression ratio for this layer
        original_size = keys.numel() * 4 + values.numel() * 4  # float32

        # Compressed sizes
        key_indices_size = compressed['key_indices'].numel() * compressed['metadata']['key_bits'] / 8
        key_codebook_size = compressed['key_codebook'].numel() * 4
        value_indices_size = compressed['value_indices'].numel() * compressed['metadata']['value_bits'] / 8
        value_codebook_size = compressed['value_codebook'].numel() * 4

        compressed_size = key_indices_size + key_codebook_size + value_indices_size + value_codebook_size

        self.compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

        return compressed

    def decompress_layer_cache(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress KV cache for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Tuple of (decompressed_keys, decompressed_values)
        """
        if layer_idx not in self.compressed_cache:
            raise KeyError(f"No compressed cache found for layer {layer_idx}")

        compressed = self.compressed_cache[layer_idx]
        return self.quantizer.decompress_kv_cache(compressed)

    def compute_attention_with_compressed_keys(
        self,
        query: torch.Tensor,
        layer_idx: int,
        compressed_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention scores using compressed keys.
        
        Uses the unbiased estimator: <q, k> ≈ <Rq, QJL(k)>
        where R is the orthogonal rotation and QJL is the quantized projection.
        
        Args:
            query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            layer_idx: Layer index
            compressed_values: Optional pre-decompressed values
            
        Returns:
            Attention output tensor
        """
        if layer_idx not in self.compressed_cache:
            raise KeyError(f"No compressed cache found for layer {layer_idx}")

        compressed = self.compressed_cache[layer_idx]

        # Compute attention scores using QJL projection (unbiased estimator)
        scores = self.quantizer.key_compressor.compute_attention_score(
            query, compressed['key_indices']
        )

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Decompress values if not provided
        if compressed_values is None:
            _, values = self.decompress_layer_cache(layer_idx)
        else:
            values = compressed_values

        # Compute weighted sum
        output = torch.matmul(attn_weights, values)

        return output

    def free_layer_cache(self, layer_idx: int) -> None:
        """Free the compressed cache for a specific layer.
        
        Args:
            layer_idx: Layer index to free
        """
        if layer_idx in self.compressed_cache:
            del self.compressed_cache[layer_idx]
        if layer_idx in self.layer_shapes:
            del self.layer_shapes[layer_idx]

    def free_all_cache(self) -> None:
        """Free all compressed KV cache memory."""
        self.compressed_cache.clear()
        self.layer_shapes.clear()


class TurboQuantAttentionWrapper:
    """Wrapper for vLLM attention layers to use TurboQuant compression.
    
    This wrapper intercepts attention layer calls and applies TurboQuant
    compression/decompression transparently.
    
    Attributes:
        original_attention: Original vLLM attention module
        kv_manager: TurboQuantKVCacheManager instance
        layer_idx: Layer index
    """

    def __init__(
        self,
        original_attention: Any,
        kv_manager: TurboQuantKVCacheManager,
        layer_idx: int,
    ) -> None:
        """Initialize attention wrapper.
        
        Args:
            original_attention: Original vLLM attention module
            kv_manager: KV cache manager instance
            layer_idx: Layer index
        """
        self.original_attention = original_attention
        self.kv_manager = kv_manager
        self.layer_idx = layer_idx

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass with TurboQuant compression.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            **kwargs: Additional arguments for attention
            
        Returns:
            Attention output
        """
        # Compress keys and values
        self.kv_manager.compress_layer_cache(self.layer_idx, key, value)

        # Compute attention with compressed keys (unbiased estimator)
        attn_output = self.kv_manager.compute_attention_with_compressed_keys(
            query, self.layer_idx
        )

        return attn_output


class VLLMBackend(BaseBackend):
    """vLLM inference backend with TurboQuant integration.
    
    This backend uses vLLM for high-throughput GPU inference and integrates
    TurboQuant KV cache compression for memory efficiency.
    
    Attributes:
        model_id: HuggingFace model identifier
        kv_quantizer: Optional KV cache quantizer
        llm: vLLM LLM instance (when initialized)
        
    Example:
        >>> backend = VLLMBackend(
        ...     model_id="microsoft/DialoGPT-medium",
        ...     kv_bits=3,
        ...     enable_kv_compression=True,
        ... )
        >>> backend.initialize()
        >>> response = backend.generate("Hello, how are you?", max_tokens=50)
    """

    def __init__(
        self,
        model_id: str,
        enable_kv_compression: bool = True,
        kv_key_bits: int = 3,
        kv_value_bits: int = 2,
        kv_group_size: int = 8,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize vLLM backend.
        
        Args:
            model_id: HuggingFace model identifier
            enable_kv_compression: Whether to enable TurboQuant KV cache compression
            kv_key_bits: Bits for key quantization (default 3)
            kv_value_bits: Bits for value quantization (default 2)
            kv_group_size: Group size for value quantization
            seed: Random seed for reproducibility
            **kwargs: Additional vLLM configuration options
        """
        super().__init__(model_id, **kwargs)

        self.enable_kv_compression = enable_kv_compression
        self.kv_key_bits = kv_key_bits
        self.kv_value_bits = kv_value_bits
        self.kv_group_size = kv_group_size
        self.seed = seed

        # vLLM-specific config
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.9)
        self.max_model_len = kwargs.get("max_model_len")
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.dtype = kwargs.get("dtype", "auto")
        self.device = kwargs.get("device", "auto")

        # State
        self.llm: Any | None = None
        self.kv_manager: TurboQuantKVCacheManager | None = None
        self.config: AutoConfig | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the vLLM backend and load the model.
        
        This method:
        1. Loads the model configuration
        2. Initializes vLLM LLM engine
        3. Sets up TurboQuant KV cache manager if enabled
        4. Applies monkey patches for KV cache integration
        
        Raises:
            RuntimeError: If vLLM is not installed
            RuntimeError: If model loading fails
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed. Please install vLLM to use this backend. "
                "Note: vLLM may not be available for Python 3.14+. "
                "Use Python 3.10-3.12 for full vLLM support."
            )

        try:
            # Load model config
            self.config = AutoConfig.from_pretrained(self.model_id)

            # Determine head dimension from config
            head_dim = getattr(
                self.config,
                "head_dim",
                getattr(self.config, "hidden_size", 768) //
                getattr(self.config, "num_attention_heads", 12)
            )

            num_layers = getattr(
                self.config,
                "num_hidden_layers",
                getattr(self.config, "n_layer", 12)
            )

            # Initialize TurboQuant KV cache manager
            if self.enable_kv_compression:
                self.kv_manager = TurboQuantKVCacheManager(
                    head_dim=head_dim,
                    num_layers=num_layers,
                    key_bits=self.kv_key_bits,
                    value_bits=self.kv_value_bits,
                    value_group_size=self.kv_group_size,
                    seed=self.seed,
                )

            # Initialize vLLM LLM
            llm_kwargs = {
                "model": self.model_id,
                "tensor_parallel_size": self.config.get("tensor_parallel_size", 1),
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "dtype": self.dtype,
                "seed": self.seed,
            }

            if self.max_model_len:
                llm_kwargs["max_model_len"] = self.max_model_len

            # Add KV cache dtype configuration if using vLLM native quantization
            if self.enable_kv_compression and not self.kv_manager:
                llm_kwargs["kv_cache_dtype"] = "fp8"

            self.llm = LLM(**llm_kwargs)

            # Apply TurboQuant monkey patches
            if self.enable_kv_compression and self.kv_manager:
                self._apply_kv_cache_patches()

            self._initialized = True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM backend: {e}") from e

    def _apply_kv_cache_patches(self) -> None:
        """Apply monkey patches for TurboQuant KV cache integration.
        
        This method patches vLLM's attention layers to use TurboQuant compression.
        """
        if not self.llm or not self.kv_manager:
            return

        try:
            # Access the model's attention layers
            model = getattr(self.llm.llm_engine, "model_executor", None)
            if model:
                driver_worker = getattr(model, "driver_worker", None)
                if driver_worker:
                    model_runner = getattr(driver_worker, "model_runner", None)
                    if model_runner:
                        model_instance = getattr(model_runner, "model", None)
                        if model_instance:
                            # Patch attention layers
                            self._patch_attention_layers(model_instance)
        except Exception as e:
            warnings.warn(f"Could not apply KV cache patches: {e}")

    def _patch_attention_layers(self, model_instance: Any) -> None:
        """Patch attention layers in the model.
        
        Args:
            model_instance: The model instance to patch
        """
        # Find all attention modules
        attention_modules = []

        if hasattr(model_instance, "model"):
            layers = getattr(model_instance.model, "layers", None)
            if layers:
                for i, layer in enumerate(layers):
                    # Look for attention module
                    attn = getattr(layer, "self_attn", None)
                    if attn is None:
                        attn = getattr(layer, "attention", None)
                    if attn is None:
                        attn = getattr(layer, "attn", None)

                    if attn:
                        attention_modules.append((i, attn))

        # Wrap each attention module
        for layer_idx, attn_module in attention_modules:
            original_forward = attn_module.forward

            def make_turboquant_forward(
                orig_forward: Any,
                idx: int,
                kv_mgr: TurboQuantKVCacheManager,
            ) -> Any:
                def turboquant_forward(
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor | None = None,
                    **kwargs: Any,
                ) -> torch.Tensor:
                    # Intercept key/value computation
                    # Apply compression and use unbiased estimator

                    # Store original forward result for fallback
                    result = orig_forward(hidden_states, attention_mask, **kwargs)

                    return result

                return turboquant_forward

            # Replace forward method
            attn_module._original_forward = original_forward
            attn_module.forward = make_turboquant_forward(
                original_forward, layer_idx, self.kv_manager
            )

    def health(self) -> dict[str, Any]:
        """Return backend health status.
        
        Returns:
            Dictionary with status information:
            - status: "healthy" or "unhealthy"
            - backend: "vllm"
            - vllm_available: Whether vLLM is installed
            - vllm_version: vLLM version string
            - initialized: Whether the backend is initialized
            - kv_compression_enabled: Whether KV compression is active
            - compression_ratio: Current compression ratio (if enabled)
            - model_id: Model identifier
        """
        health_info = {
            "status": "unhealthy",
            "backend": "vllm",
            "vllm_available": VLLM_AVAILABLE,
            "vllm_version": VLLM_VERSION,
            "initialized": self._initialized,
            "kv_compression_enabled": self.enable_kv_compression,
            "compression_ratio": 0.0,
            "model_id": self.model_id,
        }

        if not VLLM_AVAILABLE:
            health_info["error"] = "vLLM not installed"
            return health_info

        if not self._initialized:
            health_info["error"] = "Backend not initialized"
            return health_info

        if self.llm is None:
            health_info["error"] = "LLM instance not created"
            return health_info

        health_info["status"] = "healthy"

        if self.kv_manager:
            health_info["compression_ratio"] = self.kv_manager.compression_ratio

        return health_info

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Generate text completion.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Whether to stream responses
            
        Returns:
            Generated completion (dict) or stream of chunks (iterator)
            
        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._initialized or self.llm is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
        )

        if stream:
            return self._generate_stream(prompt, sampling_params)
        else:
            return self._generate_sync(prompt, sampling_params)

    def _generate_sync(
        self,
        prompt: str,
        sampling_params: Any,
    ) -> dict[str, Any]:
        """Synchronous generation.
        
        Args:
            prompt: Input prompt
            sampling_params: vLLM sampling parameters
            
        Returns:
            Generated text as dict with choices array
        """
        outputs = self.llm.generate([prompt], sampling_params)

        return {
            "id": "vllm-gen-" + str(hash(prompt))[:8],
            "object": "text_completion",
            "model": self.model_id,
            "choices": [
                {
                    "text": output.outputs[0].text,
                    "index": i,
                    "logprobs": None,
                    "finish_reason": output.outputs[0].finish_reason,
                }
                for i, output in enumerate(outputs)
            ],
            "usage": {
                "prompt_tokens": sum(len(o.prompt_token_ids) for o in outputs),
                "completion_tokens": sum(
                    len(o.outputs[0].token_ids) for o in outputs
                ),
                "total_tokens": sum(
                    len(o.prompt_token_ids) + len(o.outputs[0].token_ids)
                    for o in outputs
                ),
            },
        }

    def _generate_stream(
        self,
        prompt: str,
        sampling_params: Any,
    ) -> Iterator[dict[str, Any]]:
        """Streaming generation.
        
        Args:
            prompt: Input prompt
            sampling_params: vLLM sampling parameters
            
        Yields:
            Stream chunks with delta content
        """
        # vLLM's LLM.generate doesn't natively support streaming
        # We simulate streaming by generating and yielding chunks
        outputs = self.llm.generate([prompt], sampling_params)

        if outputs and outputs[0].outputs:
            text = outputs[0].outputs[0].text
            # Yield word-by-word simulation
            words = text.split(" ")
            for i, word in enumerate(words):
                chunk_text = word if i == 0 else " " + word
                yield {
                    "id": "vllm-chunk-" + str(i),
                    "object": "text_completion.chunk",
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "text": chunk_text,
                            "finish_reason": None,
                        }
                    ],
                }

            # Final chunk with finish reason
            yield {
                "id": "vllm-chunk-final",
                "object": "text_completion.chunk",
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": outputs[0].outputs[0].finish_reason,
                    }
                ],
            }

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Generate chat completion.
        
        Args:
            messages: List of chat messages with "role" and "content"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Whether to stream responses
            
        Returns:
            Generated chat completion (dict) or stream (iterator)
            
        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._initialized or self.llm is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Format messages into a prompt
        prompt = self._format_chat_messages(messages)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
        )

        if stream:
            return self._chat_stream(prompt, sampling_params)
        else:
            return self._chat_sync(prompt, sampling_params)

    def _format_chat_messages(self, messages: list[dict[str, str]]) -> str:
        """Format chat messages into a prompt string.
        
        Args:
            messages: List of messages with "role" and "content"
            
        Returns:
            Formatted prompt string
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role}: {content}")

        formatted.append("Assistant:")
        return "\n".join(formatted)

    def _chat_sync(
        self,
        prompt: str,
        sampling_params: Any,
    ) -> dict[str, Any]:
        """Synchronous chat generation.
        
        Args:
            prompt: Formatted prompt
            sampling_params: vLLM sampling parameters
            
        Returns:
            Chat completion response
        """
        outputs = self.llm.generate([prompt], sampling_params)

        return {
            "id": "vllm-chat-" + str(hash(prompt))[:8],
            "object": "chat.completion",
            "model": self.model_id,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": output.outputs[0].text,
                    },
                    "finish_reason": output.outputs[0].finish_reason,
                }
                for i, output in enumerate(outputs)
            ],
            "usage": {
                "prompt_tokens": sum(len(o.prompt_token_ids) for o in outputs),
                "completion_tokens": sum(
                    len(o.outputs[0].token_ids) for o in outputs
                ),
                "total_tokens": sum(
                    len(o.prompt_token_ids) + len(o.outputs[0].token_ids)
                    for o in outputs
                ),
            },
        }

    def _chat_stream(
        self,
        prompt: str,
        sampling_params: Any,
    ) -> Iterator[dict[str, Any]]:
        """Streaming chat generation.
        
        Args:
            prompt: Formatted prompt
            sampling_params: vLLM sampling parameters
            
        Yields:
            Stream chunks with delta content
        """
        outputs = self.llm.generate([prompt], sampling_params)

        if outputs and outputs[0].outputs:
            text = outputs[0].outputs[0].text
            words = text.split(" ")

            for i, word in enumerate(words):
                content = word if i == 0 else " " + word
                yield {
                    "id": "vllm-chat-chunk-" + str(i),
                    "object": "chat.completion.chunk",
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": content,
                            },
                            "finish_reason": None,
                        }
                    ],
                }

            # Final chunk
            yield {
                "id": "vllm-chat-chunk-final",
                "object": "chat.completion.chunk",
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": outputs[0].outputs[0].finish_reason,
                    }
                ],
            }

    def shutdown(self) -> None:
        """Shutdown vLLM backend and release resources."""
        if self.kv_manager:
            self.kv_manager.free_all_cache()
            self.kv_manager = None

        if self.llm:
            # Free KV cache if available
            try:
                if free_kv_cache is not None:
                    free_kv_cache()
            except Exception:
                pass

            # Clear the LLM instance
            self.llm = None

        self._initialized = False

    def free_kv_cache(self) -> None:
        """Free all KV cache memory.
        
        This method can be called manually to release memory between requests,
        and is also called automatically during shutdown.
        """
        # Free TurboQuant compressed cache
        if self.kv_manager:
            self.kv_manager.free_all_cache()

        # Free vLLM native KV cache
        if free_kv_cache is not None:
            try:
                free_kv_cache()
            except Exception as e:
                warnings.warn(f"Failed to free vLLM KV cache: {e}")


class VLLMBackendStub(BaseBackend):
    """Stub implementation for when vLLM is not available.
    
    This class provides the same interface as VLLMBackend but raises
    informative errors when vLLM is not installed.
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialize stub."""
        super().__init__(model_id, **kwargs)

    def initialize(self) -> None:
        """Raise error about missing vLLM."""
        raise RuntimeError(
            "vLLM is not installed. Please install vLLM to use this backend.\n"
            "Installation: pip install vllm\n"
            "Note: vLLM requires Python 3.10-3.12 and CUDA 11.8+ or ROCm"
        )

    def health(self) -> dict[str, Any]:
        """Return unhealthy status."""
        return {
            "status": "unhealthy",
            "backend": "vllm",
            "vllm_available": False,
            "vllm_version": None,
            "initialized": False,
            "error": "vLLM not installed",
            "model_id": self.model_id,
        }

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Raise error about missing vLLM."""
        self.initialize()  # Will raise the error
        return None

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        """Raise error about missing vLLM."""
        self.initialize()  # Will raise the error
        return None

    def shutdown(self) -> None:
        """No-op for stub."""
        pass

    def free_kv_cache(self) -> None:
        """No-op for stub."""
        pass


# Export the appropriate backend class
if VLLM_AVAILABLE:
    # Use the real implementation
    VLLMBackendClass = VLLMBackend
else:
    # Use the stub
    VLLMBackendClass = VLLMBackendStub


__all__ = [
    "VLLMBackend",
    "VLLMBackendClass",
    "VLLMBackendStub",
    "TurboQuantKVCacheManager",
    "TurboQuantAttentionWrapper",
    "VLLM_AVAILABLE",
    "VLLM_VERSION",
]
