"""Backend adapters for inference engines.

This module provides pluggable backends:
- vLLM: High-throughput GPU inference with TurboQuant KV cache compression
- llama.cpp: Broad hardware support with GGUF
"""

from llm_compress.backends.base import BaseBackend
from llm_compress.backends.vllm import (
    VLLM_AVAILABLE,
    VLLM_VERSION,
    VLLMBackend,
    VLLMBackendClass,
    VLLMBackendStub,
    TurboQuantAttentionWrapper,
    TurboQuantKVCacheManager,
)
from llm_compress.backends.llama_cpp import LlamaCppBackend
from llm_compress.backends.registry import get_backend, list_backends

__all__ = [
    "BaseBackend",
    "VLLMBackend",
    "VLLMBackendClass",
    "VLLMBackendStub",
    "LlamaCppBackend",
    "TurboQuantKVCacheManager",
    "TurboQuantAttentionWrapper",
    "get_backend",
    "list_backends",
    "VLLM_AVAILABLE",
    "VLLM_VERSION",
]
