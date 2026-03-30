"""Base backend interface.

All inference backends must implement this interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class BaseBackend(ABC):
    """Abstract base class for inference backends.

    All backends (vLLM, llama.cpp) must implement this interface
    to be used by the API server.
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialize the backend.

        Args:
            model_id: HuggingFace model identifier
            **kwargs: Backend-specific configuration
        """
        self.model_id = model_id
        self.config = kwargs

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend and load the model."""
        pass

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return backend health status.

        Returns:
            Dictionary with status information
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Whether to stream responses

        Returns:
            Generated completion (dict) or stream of chunks (iterator)
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the backend and release resources."""
        pass
