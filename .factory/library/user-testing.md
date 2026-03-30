# User Testing Guide for llm-compress

**Mission:** llm-compress - LLM compression and serving tool
**Milestones:** core-python, quantization, backends, server, tui, typescript, integration, release

---

## Validation Concurrency

Machine specs: 16 cores, 27GB RAM, ~9.7GB available

| Surface | Cost per Validator | Max Concurrent | Notes |
|---------|-------------------|----------------|-------|
| CLI | ~100MB RAM, 1 CPU | 5 | Lightweight shell commands |
| API | ~500MB RAM, 2 CPU | 3 | Requires running server |
| TUI | ~300MB RAM, 1 CPU | 3 | Terminal emulation overhead |

**Total budget:**
- Available headroom: ~9.7GB
- Using 70% for safety: ~6.8GB usable
- CLI + API + TUI combined: ~900MB
- Conservative max concurrent: 3 validators total

### Recommended Grouping
1. **CLI-only features:** Run 3-5 validators in parallel
2. **API features:** Run 2-3 validators, each on different port
3. **TUI features:** Run 1 validator at a time
4. **Cross-area flows:** Run 1 validator (end-to-end)

---

## Testing Constraints

### Time Constraints

| Operation | Typical Duration |
|-----------|-----------------|
| Small model download (<1B) | 1-2 minutes |
| 7B model quantization (4-bit) | 5-10 minutes |
| Server startup | 10-30 seconds |
| API response (short) | 1-5 seconds |

**Recommendations:**
- Use tiny models (distilgpt2, DialoGPT-small) for tests
- Pre-download models to avoid network variability
- Set timeouts: 5min for downloads, 15min for quantization

### Network Constraints
- HuggingFace Hub access required for downloads
- Some models gated (require HF_TOKEN)
- Rate limiting possible

**Mitigations:**
- Use public models for tests
- Cache models between test runs
- Set HF_TOKEN for gated model tests

---

## Known Testing Gotchas

1. **Port conflicts:** Server tests must use unique ports (3200-3299 range)
2. **Cache pollution:** Tests should clean up or use isolated cache dirs
3. **Process cleanup:** Server processes must be killed after tests
4. **Terminal state:** TUI tests need proper TTY setup
5. **Model size:** Large models cause timeouts - use small ones for tests
6. **HF rate limits:** Frequent downloads may trigger rate limiting
7. **FastAPI HTTP codes:** FastAPI/Pydantic returns 422 for validation errors by convention, not 400

---

## Flow Validator Guidance: CLI Surface

### Isolation Boundaries
- Use isolated cache directories via `--cache-dir` flag
- Tests should not interfere with default `~/.cache/llm-compress/`
- Each validator should use unique temp directories

### Shared State to Avoid
- Default cache directory (`~/.cache/llm-compress/`)
- Global configuration files
- Running server processes on standard ports

### Constraints
- Clean up temp directories after tests
- Kill any server processes started during testing
- Verify exit codes and stdout/stderr content

---

## Flow Validator Guidance: API Surface

### Isolation Boundaries
- Each validator must use a unique port (3200-3299 range)
- Use `--port` flag to specify custom port
- Server must be stopped after each test using kill command

### Shared State to Avoid
- Port conflicts - verify port is free before starting
- Model cache can be shared (read-only)
- Server process state - always cleanup

### Constraints
- Wait for `/health` endpoint to return 200 before testing
- Use `lsof -ti :PORT | xargs kill -9` for cleanup
- Set reasonable timeouts (30s for startup, 10s for requests)
- Test with small models only (tiny-gpt2, DialoGPT-small)

**HTTP Status Code Notes:**
- FastAPI returns 422 for JSON validation/parse errors by convention
- 400 is typically for bad request structure
- 422 = Unprocessable Entity (validation failed)

---

## Flow Validator Guidance: Server Milestone

### Setup Requirements
1. Ensure `llm-compress` CLI is installed (`pip install -e .`)
2. Use TheBloke GGUF models for testing (e.g., TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
3. Ports 3200-3210 are reserved for server testing

### Model Setup for Testing

**For llama.cpp backend testing:**
The llama.cpp backend requires GGUF format models. Download a pre-converted GGUF model:

```bash
# Download TheBloke model to temp directory
python -c "
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    filename='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    local_dir='/tmp/gguf_models'
)
print(f'Model at: {model_path}')
"

# Serve with direct GGUF path
llm-compress serve /tmp/gguf_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --backend llama-cpp --port 3208
```

**Note:** TheBloke GGUF models work well for testing because:
- They are pre-converted to GGUF format
- No conversion tools needed
- Small size (Q4_K_M ~ 600MB) for fast testing

### Assertion Groups

**CLI Serve Commands (VAL-CLI-007 through VAL-CLI-010, VAL-CLI-016):**
- Test serve command starts server
- Test different backends (vllm, llama-cpp)
- Test custom port configuration
- Each test uses unique port

**API Endpoints (VAL-API-001 through VAL-API-014):**
- Start server on specific port
- Test all API endpoints
- Verify JSON response formats
- Stop server after tests

### Evidence Collection
- Save curl responses as JSON files
- Capture server logs
- Record port numbers used
- Document any failures with exact error messages

### Cleanup Requirements
- Kill all server processes after testing
- Remove any temp cache directories
- Verify ports are released

---

## Test Model Recommendations

For fast, reliable tests:

| Model | Size | Use Case |
|-------|------|----------|
| microsoft/DialoGPT-small | 117M | Download, basic inference |
| sshleifer/tiny-gpt2 | 17M | Very fast tests |
| distilbert-base-uncased | 66M | Classification tests |
| TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF | ~600MB | llama.cpp backend tests |

---

## Environment Setup Checklist

Before running validators:
- [ ] Python environment set up
- [ ] Package installed (pip install -e .)
- [ ] HF_TOKEN set (for gated models)
- [ ] Ports 3200-3299 available
- [ ] Cache directory writeable
- [ ] Network access to HuggingFace
- [ ] GPU available (optional)
