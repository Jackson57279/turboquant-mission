"""Backend registry.

This module provides backend discovery and instantiation.
"""

from typing import Any
from llm_compress.backends.base import BaseBackend
from llm_compress.backends.vllm import VLLMBackend
from llm_compress.backends.llama_cpp import LlamaCppBackend


_BACKENDS: dict[str, type[BaseBackend]] = {
    "vllm": VLLMBackend,
    "llama-cpp": LlamaCppBackend,
}


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
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}")
    
    backend_class = _BACKENDS[name]
    return backend_class(model_id, **kwargs)


def list_backends() -> list[str]:
    """List available backend names.
    
    Returns:
        List of backend names
    """
    return list(_BACKENDS.keys())
