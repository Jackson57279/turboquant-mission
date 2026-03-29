# LLM Compress

Unified LLM Quantization & Inference System combining AirLLM's layer-wise weight quantization with TurboQuant's KV cache compression.

## Features

- **Weight Quantization**: 4-bit and 8-bit block-wise quantization for model weights
- **KV Cache Compression**: 3-bit keys + 2-bit values via TurboQuant-style compression
- **Layer-wise Loading**: AirLLM-style on-demand layer loading for ultra-low memory inference
- **Pluggable Backends**: vLLM (high-throughput) and llama.cpp (broad hardware support)
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API

## Installation

```bash
pip install llm-compress
```

Or install from source:

```bash
git clone https://github.com/llm-compress/llm-compress.git
cd llm-compress/python
pip install -e ".[dev]"
```

## Quick Start

### Download a Model

```bash
llm-compress download meta-llama/Llama-2-7b-hf
```

### Quantize the Model

```bash
# 4-bit quantization
llm-compress quantize meta-llama/Llama-2-7b-hf --bits 4

# 4-bit with KV cache compression
llm-compress quantize meta-llama/Llama-2-7b-hf --bits 4 --kv-cache
```

### Start the API Server

```bash
llm-compress serve meta-llama/Llama-2-7b-hf --port 3200
```

### List Downloaded Models

```bash
llm-compress list
```

### Launch the TUI

```bash
llm-compress tui
```

## Python API

```python
from llm_compress import quantize_model, get_backend

# Quantize a model
model = quantize_model("meta-llama/Llama-2-7b-hf", bits=4)

# Get a backend
backend = get_backend("vllm", "meta-llama/Llama-2-7b-hf")
backend.initialize()

# Generate text
result = backend.generate("Hello, world!", max_tokens=50)
print(result)
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 12.8+ (optional, for GPU acceleration)

## License

MIT License
