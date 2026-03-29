# vLLM Backend Adapter with TurboQuant KV Cache Integration

## Overview

The vLLM backend adapter provides integration between the TurboQuant KV cache compression system and vLLM's high-throughput inference engine. This implementation supports both Python 3.10-3.12 (with vLLM installed) and Python 3.14+ (using a stub that raises informative errors).

## Key Components

### 1. TurboQuantKVCacheManager

The `TurboQuantKVCacheManager` class manages the TurboQuant compression for vLLM's KV cache:

```python
from llm_compress.backends.vllm import TurboQuantKVCacheManager

manager = TurboQuantKVCacheManager(
    head_dim=64,           # Dimension per attention head
    num_layers=12,         # Number of transformer layers
    key_bits=3,            # Bits for key quantization
    value_bits=2,          # Bits for value quantization
    key_proj_dim=32,       # Projected dimension for keys (default: head_dim//2)
    value_group_size=8,    # Group size for value quantization
    seed=42,               # Random seed for reproducibility
)
```

**Features:**
- Automatic fitting on first use if not pre-fitted
- Per-layer cache compression/decompression
- Compression ratio tracking
- Attention computation with compressed keys using unbiased estimator
- Memory management with `free_layer_cache()` and `free_all_cache()`

### 2. VLLMBackend

The main backend class that integrates with vLLM:

```python
from llm_compress.backends.vllm import VLLMBackend

backend = VLLMBackend(
    model_id="microsoft/DialoGPT-medium",
    enable_kv_compression=True,
    kv_key_bits=3,
    kv_value_bits=2,
    kv_group_size=8,
)

backend.initialize()
response = backend.generate("Hello, how are you?", max_tokens=50)
```

**Configuration Options:**
- `enable_kv_compression`: Enable TurboQuant KV cache compression (default: True)
- `kv_key_bits`: Bits for key quantization (default: 3)
- `kv_value_bits`: Bits for value quantization (default: 2)
- `kv_group_size`: Group size for value quantization (default: 8)
- `temperature`, `top_p`: Sampling parameters
- `max_model_len`: Maximum model length
- `gpu_memory_utilization`: GPU memory fraction to use
- `dtype`: Data type ("auto", "float16", "bfloat16", etc.)

### 3. VLLMBackendStub

When vLLM is not installed (Python 3.14+), the stub provides the same interface but raises informative errors:

```python
from llm_compress.backends.vllm import VLLMBackendStub

stub = VLLMBackendStub("test-model")
stub.health()  # Returns unhealthy status with error message
stub.initialize()  # Raises RuntimeError with installation instructions
```

## Usage via Registry

The backend is registered in the backend registry:

```python
from llm_compress.backends import get_backend, list_backends

# List available backends
backends = list_backends()  # ["vllm", "llama-cpp"]

# Get backend instance
backend = get_backend("vllm", "microsoft/DialoGPT-medium")
backend.initialize()
```

## KV Cache Integration Architecture

### Compression Flow

1. **During Inference**: When vLLM processes a request, the attention layers generate key and value tensors
2. **Compression**: `TurboQuantKVCacheManager.compress_layer_cache()` compresses KV tensors:
   - Keys: Orthogonal rotation → QJL projection → Lloyd-Max quantization (3-bit)
   - Values: Group quantization using shared codebook (2-bit)
3. **Storage**: Compressed indices and codebooks stored per layer
4. **Decompression**: `decompress_layer_cache()` reconstructs tensors when needed

### Attention with Compressed Keys

The unbiased estimator for attention scores:

```python
# Original: score = <query, key>
# With compression: score ≈ <R(query), QJL(key)>
# where R is orthogonal rotation and QJL is quantized projection

scores = backend.kv_manager.compute_attention_with_compressed_keys(
    query, layer_idx=0
)
```

### Memory Management

```python
# Free specific layer cache
backend.kv_manager.free_layer_cache(5)

# Free all compressed cache
backend.kv_manager.free_all_cache()

# Free vLLM native KV cache (if available)
backend.free_kv_cache()

# Shutdown backend and release all resources
backend.shutdown()
```

## Health Check

The health endpoint provides status information:

```python
health = backend.health()
# {
#     "status": "healthy",
#     "backend": "vllm",
#     "vllm_available": True,
#     "vllm_version": "0.18.0",
#     "initialized": True,
#     "kv_compression_enabled": True,
#     "compression_ratio": 3.76,
#     "model_id": "microsoft/DialoGPT-medium"
# }
```

## Error Handling

### Missing vLLM

When vLLM is not installed:

```python
from llm_compress.backends.vllm import VLLM_AVAILABLE, VLLMBackend

if not VLLM_AVAILABLE:
    print("vLLM not installed - using stub backend")
    # Backend operations will raise RuntimeError with installation instructions
```

### Model Loading Errors

```python
try:
    backend.initialize()
except RuntimeError as e:
    # Handle initialization failure
    print(f"Failed to initialize: {e}")
```

## Testing

### Unit Tests

The test suite covers:

```bash
# Run vLLM backend tests
pytest tests/test_vllm_backend.py -v

# Run with specific test class
pytest tests/test_vllm_backend.py::TestTurboQuantKVCacheManager -v

# Run stub tests only (no vLLM required)
pytest tests/test_vllm_backend.py::TestVLLMBackendStub -v
```

### Test Coverage

- Manager initialization and fitting
- Compression/decompression round-trip
- Attention computation with compressed keys
- Memory management (free cache)
- Health check functionality
- Backend initialization with mocked vLLM
- Error handling for missing vLLM
- Stub backend behavior

## References

- TurboQuant Paper: "Making KV Cache Compression Robust to Pruning for Efficient LLM Inference"
- vLLM Documentation: https://docs.vllm.ai/
- vLLM KV Cache Quantization: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/

## Implementation Notes

### Monkey Patching

The backend attempts to patch vLLM's attention layers for transparent integration:

1. Access the model through `llm.llm_engine.model_executor.driver_worker.model_runner.model`
2. Find attention modules in transformer layers
3. Wrap forward methods with compression/decompression logic

Note: This is an experimental feature and may not work with all model architectures.

### Python 3.14+ Compatibility

vLLM is not available for Python 3.14+. The adapter handles this by:

1. Conditional import with try/except
2. `VLLM_AVAILABLE` flag for runtime checks
3. `VLLMBackendStub` as fallback implementation

### Performance Considerations

- Compression adds overhead (~10-20% depending on hardware)
- Decompression is faster than recomputation for long sequences
- Best suited for scenarios with limited GPU memory
- Compression ratio typically 3-4x for 3-bit keys + 2-bit values

## Future Enhancements

- Native vLLM plugin integration (when plugin API stabilizes)
- Triton kernels for GPU-accelerated compression/decompression
- Streaming KV cache compression for long contexts
- Dynamic bit-width adjustment based on available memory
