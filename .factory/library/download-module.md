# Model Download Module

## Overview

The `download` module provides functionality for downloading models from HuggingFace Hub with the following features:

- Progress bar display during download
- Custom cache directory support (default: `~/.cache/llm-compress/`)
- Metadata saving alongside model files
- Clear error messages for invalid model IDs
- Authentication support for gated models

## Usage

### Python API

```python
from llm_compress.download import download_model

# Download a model with default cache directory
model_path = download_model("microsoft/DialoGPT-medium")

# Download with custom cache directory
model_path = download_model(
    "microsoft/DialoGPT-medium",
    cache_dir="/custom/cache/path"
)

# Download with authentication token
model_path = download_model(
    "private/model",
    token="hf_xxx"
)
```

### CLI

```bash
# Download with default cache
llm-compress download microsoft/DialoGPT-medium

# Download with custom cache directory
llm-compress download microsoft/DialoGPT-medium --cache-dir /tmp/test-cache

# Download with authentication
llm-compress download private/model --token hf_xxx
```

## Cache Directory Structure

```
~/.cache/llm-compress/
├── microsoft--DialoGPT-medium/
│   ├── metadata.json
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── org--model-name/
    ├── metadata.json
    └── ...
```

## Error Handling

The module provides clear error messages:

- **Model not found**: "Model 'invalid/model' not found on HuggingFace Hub. Please check the model ID and ensure it's correct."
- **Authentication required**: "Model 'private/model' requires authentication. Please provide a valid HuggingFace token."

## Environment Variables

- `LLM_COMPRESS_CACHE_DIR`: Override default cache directory
- `HF_TOKEN`: HuggingFace token (also used by huggingface_hub)
