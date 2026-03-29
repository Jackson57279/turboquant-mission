"""Backend adapters for inference engines.

This module provides pluggable backends:
- vLLM: High-throughput GPU inference
- llama.cpp: Broad hardware support with GGUF
"""

from llm_compress.backends.base import BaseBackend
from llm_compress.backends.vllm import VLLMBackend
from llm_compress.backends.llama_cpp import LlamaCppBackend
from llm_compress.backends.registry import get_backend

__all__ = [
    "BaseBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    "get_backend",
]
