---
name: typescript-core-worker
description: TypeScript quantization implementation and NPM package structure
---

# typescript-core-worker

## When to Use This Skill

Use this skill for:
- Creating TypeScript package structure
- Implementing quantization algorithms in TypeScript
- Building WASM modules for performance-critical operations
- Setting up build configuration (TypeScript, Bun)

## Required Skills

- **None** - Standard TypeScript tools only

## Work Procedure

1. **Read mission context**
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/mission.md
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/AGENTS.md
   - Check Python implementation for algorithm details

2. **Set up package structure**
   - Create package.json with dependencies
   - Configure TypeScript (tsconfig.json)
   - Set up build scripts (Bun-based)
   - Create src/ directory structure

3. **Write tests first (TDD)**
   - Use Bun's test runner
   - Tests should verify correctness against Python reference

4. **Implement TypeScript version**
   - Port algorithms from Python
   - Use typed arrays for tensor operations
   - Consider WASM for heavy computations
   - Document differences from Python implementation

5. **Verify with tests**
   - Run bun test until all tests pass
   - Compare accuracy with Python implementation

6. **Build and package**
   - Run bun run build
   - Verify output files
   - Test CLI binary

## Example Handoff

```json
{
  "salientSummary": "Created TypeScript package structure and implemented 4-bit/8-bit weight quantization. Build outputs both CommonJS and ESM. All tests pass, achieving same accuracy as Python implementation.",
  "whatWasImplemented": "Created typescript/package.json with @opentui/core, commander.js, axios dependencies. Implemented src/quantization/weight.ts with block-wise 4-bit and 8-bit quantization. Uses Float32Array for tensor storage. Build configured for both CJS and ESM output.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "bun install", "exitCode": 0, "observation": "All dependencies installed"},
      {"command": "bun run build", "exitCode": 0, "observation": "Build succeeds, dist/ contains cjs and esm outputs"},
      {"command": "bun test", "exitCode": 0, "observation": "12 tests passed"}
    ],
    "testsAdded": [
      {"file": "tests/quantization.test.ts", "cases": [
        {"name": "test_4bit_quantization", "verifies": "4-bit quantization accuracy matches Python"},
        {"name": "test_8bit_quantization", "verifies": "8-bit quantization accuracy matches Python"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Algorithm too complex for TypeScript - needs WASM but WASM build system not ready
- Performance unacceptable compared to Python - need different approach
- OpenTUI dependency issues (Zig not installed, Bun issues)
