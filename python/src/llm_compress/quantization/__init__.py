"""Quantization module for weight and KV cache compression.

This module provides:
- Weight quantization (4-bit, 8-bit block-wise)
- KV cache quantization (3-bit keys, 2-bit values via TurboQuant)
- Layer-wise loading for low-memory inference
"""

from llm_compress.quantization.weight import quantize_model
from llm_compress.quantization.kv_cache import KVCacheQuantizer

__all__ = [
    "quantize_model",
    "KVCacheQuantizer",
]
