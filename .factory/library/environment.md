# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Required Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `HF_TOKEN` | HuggingFace API token for gated models | No | - |
| `CUDA_HOME` | CUDA installation path (if non-standard) | No | /usr/local/cuda |
| `LLM_COMPRESS_CACHE_DIR` | Model cache directory | No | ~/.cache/llm-compress |
| `LLM_COMPRESS_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No | INFO |

## Optional Variables

| Variable | Description |
|----------|-------------|
| `TRANSFORMERS_CACHE` | HuggingFace Transformers cache (we override this) |
| `HF_HUB_CACHE` | HuggingFace Hub cache (we override this) |

## External Dependencies

### HuggingFace Hub
- Used for model downloads
- API token required for gated models (Llama, etc.)
- Get token: https://huggingface.co/settings/tokens

### CUDA (Optional)
- Required for GPU acceleration
- Version: 12.8+ recommended
- Can work without CUDA (CPU-only mode)

## Platform-Specific Notes

### Linux
- Standard installation path works
- May need `sudo` for Zig installation to /usr/local/bin

### macOS
- Only Apple Silicon supported for some backends
- Install python-native: see https://stackoverflow.com/a/65432861/21230266
- MLX framework optional for Apple Silicon optimization

### Windows
- WSL2 recommended for full functionality
- Native Windows support limited for some features
