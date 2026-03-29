---
name: python-server-worker
description: API server, CLI commands, and user-facing interfaces
---

# python-server-worker

## When to Use This Skill

Use this skill for:
- Implementing CLI commands (download, quantize, serve, list, remove)
- Building FastAPI server with OpenAI-compatible endpoints
- Creating request/response validation
- Implementing server lifecycle management
- End-to-end integration tests

## Required Skills

- **None** - Standard Python tools only

## Work Procedure

1. **Read mission context**
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/mission.md
   - Read /home/dih/.factory/missions/07fa55b8-a7aa-40b1-89fb-dfb79186e53a/AGENTS.md
   - Check validation contract for assertion IDs this feature fulfills

2. **Design CLI command**
   - Define arguments and options using Click decorators
   - Plan help text and documentation
   - Consider common error cases

3. **Write tests first (TDD)**
   - For CLI: use click.testing.CliRunner
   - For API: use fastapi.testclient.TestClient
   - Tests should cover:
     - Success paths
     - Error paths (invalid args, missing files)
     - Edge cases

4. **Implement the feature**
   - For CLI: Use Click for argument parsing, rich for output formatting
   - For API: Use FastAPI, Pydantic models for validation
   - Add progress bars for long operations (download, quantize)
   - Use async/await for I/O operations
   - Handle errors gracefully with clear messages

5. **Verify with tests**
   - Run pytest until all tests pass
   - For CLI features: test with real shell commands
   - For API: test with curl

6. **Manual verification**
   - Run the actual CLI command
   - Verify behavior matches expected
   - Check output formatting

## Example Handoff

```json
{
  "salientSummary": "Implemented llm-compress serve command with --backend, --port, and --host options. Server starts successfully on specified port and responds to /health. Tested with both vLLM and llama.cpp backends.",
  "whatWasImplemented": "Created llm_compress/cli.py serve command. Supports --backend (vllm/llama-cpp), --port (default 3200), --host (default 127.0.0.1). Integrates with backend adapters to start inference server. Shows warning for unquantized models. Graceful shutdown on SIGINT.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "llm-compress serve microsoft/DialoGPT-medium --backend vllm --port 3200", "exitCode": 0, "observation": "Server started, logs show vLLM backend initialized"},
      {"command": "curl http://localhost:3200/health", "exitCode": 0, "observation": "{\"status\": \"healthy\"}"},
      {"command": "llm-compress serve unquantized-model", "exitCode": 0, "observation": "Warning displayed: 'Model not quantized. Performance may be reduced.'"}
    ],
    "testsAdded": [
      {"file": "tests/cli/test_serve.py", "cases": [
        {"name": "test_serve_starts_server", "verifies": "Server starts and responds to health check"},
        {"name": "test_serve_custom_port", "verifies": "Server listens on specified port"},
        {"name": "test_serve_unquantized_warning", "verifies": "Warning shown for unquantized models"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Backend adapter needed is not yet implemented
- Port conflict cannot be resolved (all ports in range 3200-3299 in use)
- CLI command depends on API endpoints that don't exist yet
- User input reveals ambiguous requirements
