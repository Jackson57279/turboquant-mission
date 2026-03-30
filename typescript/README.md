# llm-compress TypeScript Implementation

This directory contains the TypeScript implementation of llm-compress, a unified LLM quantization and inference system.

## Features

- **Weight Quantization**: Block-wise 4-bit and 8-bit quantization
- **KV Cache Compression**: 3-bit keys + 2-bit values (TurboQuant-style)
- **Layer-wise Loading**: On-demand layer loading for low-memory inference
- **OpenTUI-based TUI**: Interactive terminal user interface
- **OpenAI-compatible API**: HTTP API server for inference

## Installation

```bash
# Using Bun (recommended)
bun install

# Using npm
npm install
```

## Building

```bash
# Build both ESM and CommonJS outputs
bun run build

# Build ESM only
bun run build:esm

# Build CommonJS only
bun run build:cjs
```

## CLI Usage

```bash
# Download a model
llm-compress download <model_id>

# Quantize a model
llm-compress quantize <model_id> --bits 4

# Serve a model
llm-compress serve <model_id> --port 3200

# List downloaded models
llm-compress list

# Remove a model
llm-compress remove <model_id>

# Launch TUI
llm-compress tui
```

## Testing

```bash
bun test
```

## Project Structure

```
typescript/
├── src/
│   ├── quantization/     # Weight & KV cache quantization
│   ├── backends/         # vLLM & llama.cpp adapters
│   ├── server/           # OpenAI-compatible API
│   ├── tui/              # OpenTUI components
│   ├── cli.ts            # CLI entry point
│   └── index.ts          # Library exports
├── tests/                # Test files
├── package.json          # Package configuration
├── tsconfig.json         # TypeScript configuration
└── README.md             # This file
```

## License

MIT
