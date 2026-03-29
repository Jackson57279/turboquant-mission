# Architecture

How the llm-compress system works — components, relationships, data flows, invariants.

**What belongs here:** High-level architecture, component interactions, data flows, key design decisions.
**What does NOT belong here:** Implementation details, API docs (those go in docs/), low-level algorithms (document in code).

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │   CLI    │  │   TUI    │  │   API    │  │  NPM Package    │  │
│  │(Python)  │  │(OpenTUI) │  │(FastAPI) │  │  (TypeScript)   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────────────────┘  │
└───────┼─────────────┼─────────────┼─────────────────────────────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
        ┌─────────────┴─────────────┐
        │      Core Quantization      │
        │  ┌───────────────────────┐  │
        │  │   Weight Quantization │  │
        │  │   (AirLLM-style)      │  │
        │  │   - 4-bit block-wise  │  │
        │  │   - 8-bit block-wise  │  │
        │  │   - Layer-wise loading│  │
        │  └───────────────────────┘  │
        │  ┌───────────────────────┐  │
        │  │   KV Cache Quantization │  │
        │  │   (TurboQuant-style)    │  │
        │  │   - 3-bit keys (QJL)  │  │
        │  │   - 2-bit values       │  │
        │  │   - Lloyd-Max codebooks│  │
        │  └───────────────────────┘  │
        └─────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │      Backend Adapters       │
        │  ┌──────────┐  ┌──────────┐  │
        │  │  vLLM    │  │ llama.cpp│  │
        │  │ Adapter  │  │ Adapter  │  │
        │  └──────────┘  └──────────┘  │
        └─────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │      Model Storage          │
        │  ~/.cache/llm-compress/     │
        │  - Original models (HF)   │
        │  - Quantized weights       │
        │  - Metadata files          │
        └─────────────────────────────┘
```

## Component Details

### CLI (Python)
Entry point for all user commands. Built with Click.
- `download`: Fetch models from HuggingFace
- `quantize`: Apply weight/KV quantization
- `serve`: Start inference API server
- `list`/`remove`: Manage cached models
- `tui`: Launch terminal UI

### TUI (TypeScript/OpenTUI)
Interactive terminal interface. Built with @opentui/react.
- Model browser with navigation
- Quantization configuration screen
- Server control panel
- Chat interface for testing

### API Server (Python/FastAPI)
OpenAI-compatible REST API.
- `/v1/models` - List available models
- `/v1/chat/completions` - Chat with streaming support
- `/v1/completions` - Legacy completion endpoint
- `/health` - Server health check

### Quantization Engine (Python)
Hybrid quantization combining AirLLM + TurboQuant.

**Weight Quantization (AirLLM-style):**
- Block-wise 4-bit/8-bit quantization
- Layer-wise loading for memory efficiency
- Prefetching for performance

**KV Cache Quantization (TurboQuant-style):**
- 3-bit key compression with QJL projection
- 2-bit/4-bit value group quantization
- Lloyd-Max optimal codebooks
- Triton kernels for GPU acceleration

### Backend Adapters
Pluggable backends for inference.

**vLLM Backend:**
- High-throughput serving
- PagedAttention
- TurboQuant KV cache integration via monkey-patching

**llama.cpp Backend:**
- Broad hardware support
- GGUF format
- CPU and GPU (CUDA/Metal) support

## Data Flows

### Download Flow
```
User -> CLI download -> HuggingFace Hub -> Cache directory -> Metadata saved
```

### Quantize Flow
```
User -> CLI quantize -> Load model -> Apply quantization -> Save quantized -> Update metadata
```

### Serve Flow
```
User -> CLI serve -> Load backend -> Initialize model -> Start FastAPI -> Accept requests
```

### Inference Flow (vLLM)
```
Request -> FastAPI -> vLLM backend -> TurboQuant KV cache -> Generate -> Response
```

### Inference Flow (llama.cpp)
```
Request -> FastAPI -> llama.cpp backend -> GGUF model -> Generate -> Response
```

## Key Invariants

1. **Cache consistency**: All operations use the same cache directory structure
2. **Metadata tracking**: Every model has metadata.json with quantization status
3. **Backend isolation**: Backend adapters are swappable without changing API
4. **Quantization accuracy**: 4-bit >99%, 8-bit >99.5%, KV cache cos_sim >0.94
5. **Memory bounds**: Layer-wise loading keeps VRAM <4GB for 70B models

## Directory Structure

```
~/.cache/llm-compress/
├── models/
│   ├── org-name/
│   │   ├── model-name/
│   │   │   ├── original/          # HF downloaded files
│   │   │   ├── quantized-4bit/  # 4-bit quantized
│   │   │   ├── quantized-8bit/  # 8-bit quantized
│   │   │   └── metadata.json    # Model metadata
│   └── ...
└── tmp/                          # Temporary download space
```

## Design Decisions

1. **Hybrid approach**: Combine AirLLM (weight quant) + TurboQuant (KV cache) for maximum efficiency
2. **Pluggable backends**: Support both vLLM (speed) and llama.cpp (compatibility)
3. **OpenAI-compatible API**: Drop-in replacement for OpenAI API
4. **Dual package**: Python for full functionality, TypeScript for standalone TUI
5. **Layer-wise loading**: Enable 70B models on 4GB GPU without expensive hardware
