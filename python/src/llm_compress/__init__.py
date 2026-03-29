"""LLM Compress: Unified LLM Quantization & Inference System.

This package provides:
- Weight quantization (4-bit, 8-bit block-wise)
- KV cache compression (3-bit keys, 2-bit values via TurboQuant)
- Layer-wise loading (AirLLM-style)
- Pluggable inference backends (vLLM, llama.cpp)
- OpenAI-compatible API server

Example:
    >>> from llm_compress import quantize_model
    >>> model = quantize_model("meta-llama/Llama-2-7b", bits=4)
"""

__version__ = "0.1.0"
__author__ = "llm-compress Team"
__license__ = "MIT"

from llm_compress.backends import get_backend
from llm_compress.quantization import quantize_model

__all__ = [
    "__version__",
    "quantize_model",
    "get_backend",
]
