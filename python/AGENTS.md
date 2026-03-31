# Python Implementation

**Parent:** ../AGENTS.md  
**Language:** Python 3.10+  
**Lines:** ~6,100  
**Symbols:** 263

## OVERVIEW

PyTorch-based LLM quantization with vLLM/llama.cpp backends. Uses bitsandbytes for 4/8-bit weight quantization and custom TurboQuant-style KV cache compression.

## STRUCTURE

```
python/
├── src/llm_compress/        # Main package
│   ├── __init__.py          # Public API exports
│   ├── cli.py               # Click CLI (420 lines)
│   ├── download.py          # HuggingFace model downloader
│   ├── quantization/        # Quantization implementations
│   │   ├── __init__.py      # quantize_model entry
│   │   ├── weight.py        # 4/8-bit weight quantization
│   │   ├── kv_cache.py      # 3-bit keys, 2-bit values
│   │   └── layer_wise.py    # AirLLM-style layer loading
│   ├── backends/            # Inference backends
│   │   ├── __init__.py      # get_backend factory
│   │   ├── base.py          # BaseBackend ABC
│   │   ├── vllm.py          # vLLM adapter
│   │   ├── llama_cpp.py     # llama.cpp adapter
│   │   └── registry.py      # Backend registration
│   └── server/              # OpenAI-compatible API
│       ├── __init__.py
│       ├── app.py           # FastAPI factory
│       └── models.py        # Pydantic schemas
├── tests/                   # pytest test suite
└── pyproject.toml           # Package config
```

## CONVENTIONS

### Code Style
- **Black**: line-length 100
- **Ruff**: py310 target, rules: E,F,I,N,W,UP,B,C4,SIM
- **mypy**: relaxed (disallow_untyped_defs=false)

### Imports
```python
# Absolute from package
from llm_compress.backends import get_backend
from llm_compress.quantization import quantize_model

# External deps grouped
import torch
import click
from safetensors.torch import save_file
```

### Docstrings (Google Style)
```python
def quantize_tensor(tensor: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, ...]:
    """Quantize a single tensor to specified bit width.

    Args:
        tensor: Input tensor to quantize (must be 2D or can be reshaped to 2D)
        bits: Quantization bit width (4 or 8)

    Returns:
        Tuple of (quantized_tensor, quantization_state, original_shape)

    Raises:
        ValueError: If bits is not 4 or 8
    """
```

### Type Hints
- Use `from __future__ import annotations` for forward refs
- Prefer `|` over `Union[]` and `Optional[]`
- Use `collections.abc` imports: `Iterator`, `Callable`

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add quant scheme | `quantization/weight.py` | NF4/FP8 implementations |
| New backend | `backends/base.py` → `backends/new.py` | Implement BaseBackend |
| API endpoint | `server/app.py` | FastAPI routes |
| CLI command | `cli.py` | Click command groups |
| Model download | `download.py` | HuggingFace Hub integration |

## COMMANDS

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run CLI
llm-compress quantize meta-llama/Llama-2-7b-hf --bits 4

# Run tests
pytest

# Type check
mypy src/llm_compress

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## ANTI-PATTERNS

1. **Don't use relative imports** — Always absolute from `llm_compress`
2. **Don't commit without type hints** — At least for public APIs
3. **Don't use `torch.cuda` directly** — Backend handles device placement
4. **Don't use `print()` in library code** — Use `logging` module

## NOTES

- **src layout**: Code lives in `src/llm_compress/`, not top-level
- **CLI entry**: Defined in `pyproject.toml` `[project.scripts]`
- **Test location**: `python/tests/` (not root `/tests/`)
- **CUDA optional**: All GPU code should handle CPU fallback
