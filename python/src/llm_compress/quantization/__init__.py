"""Quantization module for weight and KV cache compression.

This module provides:
- Weight quantization (4-bit, 8-bit block-wise)
- KV cache quantization (3-bit keys, 2-bit values via TurboQuant)
- Layer-wise loading for low-memory inference
"""

from llm_compress.quantization.kv_cache import (
    GroupValueQuantizer,
    KVCacheQuantizer,
    LloydMaxQuantizer,
    OrthogonalRotation,
    QJLProjection,
    TurboQuantKeyCompressor,
    compute_cosine_similarity,
    estimate_compression_ratio,
)
from llm_compress.quantization.layer_wise import (
    DEFAULT_LAYER_CACHE_SIZE,
    DEFAULT_PREFETCH_AHEAD,
    MAX_VRAM_GB,
    LayerCache,
    LayerPrefetcher,
    LayerShardManager,
    LayerShardMetadata,
    LayerWiseInferenceEngine,
    LayerWiseLoader,
    ModelShardIndex,
    create_layerwise_loader,
    shard_model_for_layerwise_loading,
)
from llm_compress.quantization.weight import (
    dequantize_model,
    dequantize_tensor,
    estimate_accuracy_loss,
    get_compression_ratio,
    load_quantized_model,
    quantize_model,
    quantize_model_state_dict,
    quantize_tensor,
    save_quantized_model,
)

__all__ = [
    # Weight quantization
    "quantize_model",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_model_state_dict",
    "dequantize_model",
    "save_quantized_model",
    "load_quantized_model",
    "get_compression_ratio",
    "estimate_accuracy_loss",
    # KV cache quantization
    "KVCacheQuantizer",
    "LloydMaxQuantizer",
    "OrthogonalRotation",
    "QJLProjection",
    "TurboQuantKeyCompressor",
    "GroupValueQuantizer",
    "compute_cosine_similarity",
    "estimate_compression_ratio",
    # Layer-wise loading
    "LayerShardMetadata",
    "ModelShardIndex",
    "LayerShardManager",
    "LayerCache",
    "LayerPrefetcher",
    "LayerWiseLoader",
    "LayerWiseInferenceEngine",
    "shard_model_for_layerwise_loading",
    "create_layerwise_loader",
    "DEFAULT_LAYER_CACHE_SIZE",
    "DEFAULT_PREFETCH_AHEAD",
    "MAX_VRAM_GB",
]
