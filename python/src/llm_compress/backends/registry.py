"""Backend registry.

This module provides backend discovery and instantiation.
"""

from typing import Any

from llm_compress.backends.base import BaseBackend
from llm_compress.backends.llama_cpp import (
    LLAMA_CPP_AVAILABLE,
    LlamaCppBackend,
    LlamaCppBackendStub,
)
from llm_compress.backends.vllm import VLLM_AVAILABLE, VLLMBackend, VLLMBackendStub

_BACKENDS: dict[str, type[BaseBackend]] = {}

# Register vLLM backend if available
if VLLM_AVAILABLE:
    _BACKENDS["vllm"] = VLLMBackend
else:
    # Register stub that will raise informative errors
    _BACKENDS["vllm"] = VLLMBackendStub  # type: ignore

# Register llama.cpp backend if available
if LLAMA_CPP_AVAILABLE:
    _BACKENDS["llama-cpp"] = LlamaCppBackend
else:
    # Register stub that will raise informative errors
    _BACKENDS["llama-cpp"] = LlamaCppBackendStub  # type: ignore


def get_backend(name: str, model_id: str, **kwargs: Any) -> BaseBackend:
    """Get a backend instance by name.
    
    Args:
        name: Backend name ("vllm" or "llama-cpp")
        model_id: HuggingFace model identifier
        **kwargs: Backend-specific configuration
        
    Returns:
        Backend instance
        
    Raises:
        ValueError: If backend name is unknown
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
        )

    backend_class = _BACKENDS[name]
    return backend_class(model_id, **kwargs)


def list_backends() -> list[str]:
    """List available backend names.
    
    Returns:
        List of backend names
    """
    return list(_BACKENDS.keys())
