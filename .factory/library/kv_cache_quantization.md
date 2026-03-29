# KV Cache Quantization (TurboQuant-Style)

## Overview

Implemented TurboQuant-style KV cache quantization with the following components:

### Core Components

1. **LloydMaxQuantizer** (`llm_compress.quantization.kv_cache`)
   - Optimal scalar quantization using Lloyd-Max algorithm
   - Iteratively optimizes centroids and decision boundaries
   - Supports configurable bit widths (2-bit, 3-bit, 4-bit, etc.)
   - Minimizes mean squared error (MSE) for given bit budget

2. **OrthogonalRotation**
   - Random orthogonal matrix generation via QR decomposition
   - Preserves vector norms: ||Rx|| = ||x||
   - Deterministic with seed for reproducibility
   - Supports batched tensor operations

3. **QJLProjection** (Quantized Johnson-Lindenstrauss)
   - Dimensionality reduction while preserving inner products
   - Random Gaussian projection with proper scaling
   - Combined with Lloyd-Max quantization
   - Key property: E[<Qx, Qy>] ≈ <x, y>

4. **TurboQuantKeyCompressor**
   - 3-bit key compression pipeline:
     1. Orthogonal rotation (preserves norms)
     2. QJL projection (dimensionality reduction)
     3. Lloyd-Max quantization (3-bit)
   - Designed to preserve cosine similarity for attention
   - Supports attention score computation with compressed keys

5. **GroupValueQuantizer**
   - Group-based quantization for value vectors
   - 2-bit and 4-bit support
   - Shared codebook across group members
   - Handles padding for non-multiple sequence lengths

6. **KVCacheQuantizer** (Main entry point)
   - Combines key and value compression
   - End-to-end compress/decompress API
   - Attention computation with compressed KV cache
   - Compression ratio estimation utilities

### Usage Example

```python
from llm_compress.quantization.kv_cache import KVCacheQuantizer
import torch

# Initialize quantizer
quantizer = KVCacheQuantizer(
    head_dim=64,
    key_bits=3,       # 3-bit key compression
    value_bits=2,     # 2-bit value compression
    key_proj_dim=32,  # Project 64-dim to 32-dim
    seed=42
)

# Example KV cache tensors
batch_size = 2
num_heads = 4
seq_len = 100
head_dim = 64

keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
values = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Fit on sample data (or use auto-fit on first compress)
quantizer.fit(keys.flatten(0, -2), values.flatten(0, -2))

# Compress
compressed = quantizer.compress_kv_cache(keys, values)

# Decompress
recovered_keys, recovered_values = quantizer.decompress_kv_cache(compressed)

# Check compression ratio
from llm_compress.quantization.kv_cache import estimate_compression_ratio
stats = estimate_compression_ratio(head_dim=64, seq_len=seq_len)
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
```

### Test Coverage

40 unit tests covering:
- Lloyd-Max codebook generation correctness
- Orthogonal rotation norm preservation
- QJL projection inner product preservation
- 3-bit key compression (cos_sim > 0.90)
- 2-bit value compression (cos_sim > 0.90)
- Round-trip compression/decompression
- Attention score computation with compressed keys
- Edge cases (empty tensors, constant data, large/small values)

### Validation Contract Assertions

- VAL-QUANT-003: 3-bit key compression with cosine similarity validation
- VAL-QUANT-004: 2-bit value compression with cosine similarity validation
- VAL-QUANT-006: Lloyd-Max codebook MSE within theoretical bounds
- VAL-QUANT-007: Orthogonal rotation preserves vector norms
- VAL-QUANT-008: QJL projection preserves inner products
- VAL-QUANT-010: Unbiased estimator for attention scores

### References

- TurboQuant: Making KV Cache Compression Robust to Pruning for Efficient LLM Inference
- Johnson-Lindenstrauss Lemma for dimensionality reduction
- Lloyd-Max Quantization for optimal scalar quantization
