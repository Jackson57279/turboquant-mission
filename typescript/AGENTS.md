# TypeScript Implementation

**Parent:** ../AGENTS.md  
**Language:** TypeScript 5.3+  
**Runtime:** Bun 1.0+  
**Lines:** ~2,850  
**Symbols:** 102

## OVERVIEW

Bun-native LLM quantization with OpenTUI-based terminal UI. Dual ESM/CJS builds for maximum compatibility.

## STRUCTURE

```
typescript/
├── src/
│   ├── index.ts             # Library exports
│   ├── cli.ts               # Commander CLI (236 lines)
│   ├── quantization/        # Quantization module
│   │   ├── index.ts         # Module exports
│   │   ├── types.ts         # Type definitions
│   │   ├── weight.ts        # 4/8-bit quantization (308 lines)
│   │   └── kv_cache.ts      # KV cache compression (624 lines)
│   ├── backends/            # Backend abstractions
│   │   ├── index.ts         # Backend registry
│   │   └── types.ts         # Backend interfaces
│   ├── server/              # HTTP API types
│   │   ├── index.ts
│   │   └── types.ts
│   └── tui/                 # OpenTUI components (1479 lines)
├── tests/                   # bun:test suite
├── dist/                    # Build outputs (git-tracked)
└── package.json
```

## CONVENTIONS

### Module System
- **Dual builds**: ESM (`dist/esm/`) + CJS (`dist/cjs/`)
- **Imports**: Relative with `.js` extension (ESM compliance)
- **Exports**: Named exports preferred

```typescript
// Good
import type { QuantizationOptions } from './types.js';
export { quantizeTensor } from './weight.js';

// Avoid
import { something } from './types';  // Missing .js
export default quantizeTensor;        // Default export
```

### TypeScript Config
- **Target**: ES2022
- **Module**: NodeNext (for ESM/CJS dual output)
- **Strict**: true

### Documentation
```typescript
/**
 * Quantize a tensor to specified bit width.
 *
 * @param data - Input array
 * @param bits - Quantization bits (4 or 8)
 * @returns Quantized tensor with metadata
 *
 * @example
 * ```typescript
 * const quantized = quantizeTensor(data, 4);
 * console.log(quantized.metadata.absmax);
 * ```
 */
export function quantizeTensor(
  data: Float32Array,
  bits: number
): QuantizedTensor { }
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add quant algorithm | `quantization/weight.ts` | NF4 levels, absmax calc |
| KV cache ops | `quantization/kv_cache.ts` | 3-bit key, 2-bit value pack |
| New backend | `backends/types.ts` → implement | Add to `BackendType` union |
| TUI component | `tui/index.ts` | OpenTUI React components |
| CLI command | `cli.ts` | Commander subcommands |

## COMMANDS

```bash
# Install deps
bun install

# Build (ESM + CJS)
bun run build

# Run CLI
bun run src/cli.ts quantize <model> --bits 4
# Or after build:
llm-compress quantize <model> --bits 4

# Run tests
bun test

# Lint
bun run lint

# Type check
bun run typecheck
```

## ANTI-PATTERNS

1. **Don't forget `.js` extension** in imports — Required for ESM compatibility
2. **Don't use Node-only APIs** without platform check — Target both Bun and Node
3. **Don't import from `'bun:test'` in source** — Only in test files
4. **Don't modify `dist/` directly** — Always rebuild from source

## NOTES

- **Bun required**: Native `bun:test` and Bun APIs used
- **dist/ in git**: Build artifacts are tracked (unusual but intentional)
- **Missing tsconfig.*.json**: Referenced in package.json but files don't exist
- **TUI size**: `tui/index.ts` is 1479 lines — consider splitting
