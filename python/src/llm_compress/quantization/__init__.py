"""Quantization module for weight and KV cache compression.

This module provides:
- Weight quantization (4-bit, 8-bit block-wise)
- KV cache quantization (3-bit keys, 2-bit values via TurboQuant)
- Layer-wise loading for low-memory inference
"""

from llm_compress.quantization.weight import (
    quantize_model,
    quantize_tensor,
    dequantize_tensor,
    quantize_model_state_dict,
    dequantize_model,
    save_quantized_model,
    load_quantized_model,
    get_compression_ratio,
    estimate_accuracy_loss,
)
from llm_compress.quantization.kv_cache import KVCacheQuantizer

__all__ = [
    "quantize_model",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_model_state_dict",
    "dequantize_model",
    "save_quantized_model",
    "load_quantized_model",
    "get_compression_ratio",
    "estimate_accuracy_loss",
    "KVCacheQuantizer",
]
