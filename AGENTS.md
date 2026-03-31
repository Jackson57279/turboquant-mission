# LLM Compress — Project Knowledge Base

**Generated:** 2025-03-31  
**Commit:** 188b700  
**Branch:** master

## OVERVIEW

Dual-language monorepo implementing LLM quantization & inference. Combines AirLLM's layer-wise weight quantization with TurboQuant's KV cache compression. Python for ML/compute, TypeScript for CLI/TUI.

## STRUCTURE

```
./
├── python/           # Python implementation (PyTorch/vLLM)
├── typescript/       # TypeScript implementation (Bun/OpenTUI)
├── models/           # Downloaded models cache
├── tests/            # Cross-language TUI tests only
└── run_validation_tests.py  # Root-level test runner
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Weight quantization | `python/src/llm_compress/quantization/` or `typescript/src/quantization/` | 4-bit/8-bit block-wise |
| KV cache compression | `*/quantization/kv_cache.{py,ts}` | 3-bit keys, 2-bit values |
| Inference backends | `*/backends/` | vLLM, llama.cpp adapters |
| API server | `python/src/llm_compress/server/` | FastAPI, OpenAI-compatible |
| TUI | `typescript/src/tui/` | OpenTUI-based |
| CLI | `*/cli.{py,ts}` | Click (Python), Commander (TS) |

## CODE MAP

### Python (263 symbols)
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `quantize_model` | function | `quantization/__init__.py` | Main API entry |
| `get_backend` | function | `backends/__init__.py` | Backend factory |
| `BaseBackend` | ABC | `backends/base.py` | Backend interface |
| `quantize_tensor` | function | `quantization/weight.py` | 4/8-bit quantization |
| `main` | CLI | `cli.py` | Click CLI entry |

### TypeScript (102 symbols)
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `quantizeModel` | function | `quantization/weight.ts` | Main API entry |
| `getBackend` | function | `backends/index.ts` | Backend factory |
| `Backend` | interface | `backends/types.ts` | Backend contract |
| `createProgram` | function | `cli.ts` | Commander CLI setup |

## CONVENTIONS

### Python
- **Layout**: src layout (`src/llm_compress/`)
- **Formatter**: Black (line-length: 100)
- **Linter**: Ruff (target: py310)
- **Type checker**: mypy (relaxed: `disallow_untyped_defs=false`)
- **Imports**: Absolute from `llm_compress`
- **Docstrings**: Google style with type hints

### TypeScript
- **Runtime**: Bun (Node 20+ compatible)
- **Build**: Dual ESM/CJS outputs
- **Linter**: oxlint
- **Tests**: Bun native test runner (`bun:test`)
- **Imports**: Relative with `.js` extensions

## ANTI-PATTERNS

1. **No monorepo tooling** — Root has no workspace config; langs are isolated
2. **Root `tests/` confusion** — Only contains TUI shell tests; Python tests are in `python/tests/`
3. **TypeScript `dist/` in git** — Should be gitignored (build artifacts)
4. **Missing tsconfig.*.json** — Referenced in package.json but don't exist

## UNIQUE STYLES

1. **Dual implementations** — Every feature in both Python and TypeScript with parity tests
2. **Accuracy-focused testing** — SNR, cosine similarity, relative error metrics
3. **Performance benchmarks** — Tests include timing constraints (500ms, 1000ms)
4. **Scientific computing patterns** — Heavy use of Float32Array, vector math

## COMMANDS

```bash
# Python
cd python && pip install -e ".[dev]"
llm-compress --help
pytest

# TypeScript
cd typescript && bun install
bun run build
bun test
bun run src/cli.ts --help

# Validation
python run_validation_tests.py
```

## NOTES

- **No LSP workspace symbols** available — rely on grep/file reading
- **Bun required** for TypeScript (not npm-compatible for dev)
- **CUDA optional** — GPU acceleration available but not required
- **Research paper** included at root (`RESEARCH_PAPER.md`)
