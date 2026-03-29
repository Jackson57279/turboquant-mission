"""OpenAI-compatible API server.

This module provides:
- FastAPI-based HTTP server
- OpenAI-compatible endpoints (/v1/models, /v1/chat/completions)
- Streaming and non-streaming responses
- Authentication support
"""

from llm_compress.server.app import create_app

__all__ = [
    "create_app",
]
