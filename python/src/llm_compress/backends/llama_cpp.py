"""llama.cpp backend adapter.

This module provides llama.cpp integration with GGUF support.
"""

from llm_compress.backends.base import BaseBackend
from typing import Any, Iterator


class LlamaCppBackend(BaseBackend):
    """llama.cpp inference backend.
    
    This backend uses llama.cpp for broad hardware support,
    including CPU inference and GGUF quantized models.
    
    Attributes:
        model_id: HuggingFace model identifier
        gguf_path: Path to GGUF model file
    """
    
    def initialize(self) -> None:
        """Initialize llama.cpp backend."""
        raise NotImplementedError("llama.cpp backend not yet implemented")
    
    def health(self) -> dict[str, Any]:
        """Return backend health status."""
        raise NotImplementedError("llama.cpp backend not yet implemented")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Generate text completion."""
        raise NotImplementedError("llama.cpp backend not yet implemented")
    
    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Generate chat completion."""
        raise NotImplementedError("llama.cpp backend not yet implemented")
    
    def shutdown(self) -> None:
        """Shutdown llama.cpp backend."""
        pass
