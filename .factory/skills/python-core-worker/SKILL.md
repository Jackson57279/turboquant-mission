---
name: python-core-worker
description: Core quantization algorithms, backend adapters, and low-level LLM operations
---

# python-core-worker

## When to Use This Skill

Use this skill for:
- Implementing quantization algorithms (weight quantization, KV cache compression)
- Building backend adapters (vLLM, llama.cpp)
- Implementing layer-wise loading mechanisms
- Writing low-level tensor operations
- Creating benchmark and profiling tools

## Required Skills

- **None** - Standard Python tools only

## Work Procedure

1. **Read mission context**
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/mission.md
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/AGENTS.md
   - Check existing code in the repo to understand patterns

2. **Design and plan**
   - Identify the mathematical operations needed
   - Plan the API/interface for the module
   - Consider GPU/CPU compatibility

3. **Write tests first (TDD)**
   - Create test file with failing tests
   - Tests should cover:
     - Correctness (accuracy metrics)
     - Edge cases (empty tensors, large values)
     - Performance (timing benchmarks)
     - Error handling

4. **Implement the algorithm**
   - Use type hints throughout
   - Document with Google-style docstrings
   - Include references to papers when applicable (TurboQuant, AirLLM)
   - Add inline comments for complex operations

5. **Verify with tests**
   - Run pytest until all tests pass
   - Check coverage with pytest-cov
   - Verify accuracy metrics meet thresholds:
     - 4-bit: >99% accuracy
     - 8-bit: >99.5% accuracy
     - KV cache cos_sim >0.94 (2-bit), >0.99 (3-bit keys)

6. **Benchmark**
   - Run benchmarks to verify performance claims
   - Document results in handoff

7. **Integration check**
   - Verify the module integrates with existing code
   - Check imports work correctly

## Example Handoff

```json
{
  "salientSummary": "Implemented TurboQuant-style KV cache quantization with 3-bit keys and 2-bit values. All unit tests pass, achieving cos_sim 0.995 for keys and 0.945 for values. Benchmark shows 5x memory reduction.",
  "whatWasImplemented": "Created llm_compress/quantization/kv_cache.py with TurboQuantMSE and TurboQuantProd classes. Implements Lloyd-Max optimal scalar quantization, random orthogonal rotation, QJL projection, and group quantization for values. Includes 3 fused Triton kernels for GPU acceleration.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "pytest tests/quantization/test_kv_cache.py -v", "exitCode": 0, "observation": "7 tests passed, including cos_sim checks, unbiased estimator, and round-trip accuracy"},
      {"command": "python benchmarks/kv_cache_benchmark.py", "exitCode": 0, "observation": "5.1x memory reduction, 1.2x speedup over baseline"}
    ],
    "testsAdded": [
      {"file": "tests/quantization/test_kv_cache.py", "cases": [
        {"name": "test_3bit_key_compression", "verifies": "3-bit key compression achieves cos_sim >0.99"},
        {"name": "test_2bit_value_compression", "verifies": "2-bit value compression achieves cos_sim >0.94"},
        {"name": "test_lloyd_max_codebook", "verifies": "Lloyd-Max codebook MSE within theoretical bounds"},
        {"name": "test_unbiased_estimator", "verifies": "E[estimated] = true within 0.1% tolerance"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- GPU is required but not available and CPU fallback would significantly change the approach
- Mathematical algorithm is unclear or needs research paper clarification
- Integration with existing code reveals architectural issues
- Performance targets cannot be met with current approach
