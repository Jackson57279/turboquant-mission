# User Testing

Testing surface, required testing skills/tools, and resource cost classification per surface.

**What belongs here:** Validation surface details, testing approaches, constraints, resource costs.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Validation Surfaces

### 1. CLI Surface

**Description:** Command-line interface with commands: download, quantize, serve, list, remove, tui

**Testing approach:**
- Execute shell commands and verify exit codes, stdout, stderr
- Check file system changes (model downloads, quantized files)
- Verify process lifecycle (server start/stop)

**Tools:**
- Shell command execution
- File system checks
- curl for API verification after serve

**Required environment:**
- Python environment with package installed
- Cache directory writeable
- Network access for HuggingFace

### 2. API Surface

**Description:** OpenAI-compatible HTTP API served on port 3200+

**Testing approach:**
- HTTP requests with curl or httpx
- Validate JSON request/response formats
- Test streaming vs non-streaming responses
- Verify error status codes

**Tools:**
- curl for simple requests
- httpx for programmatic testing
- JSON validation

**Required environment:**
- Server running on known port
- Model loaded and ready
- Network access (localhost)

### 3. TUI Surface

**Description:** Terminal user interface built with OpenTUI

**Testing approach:**
- Automated with tuistory
- Screenshot capture at each step
- Keyboard input simulation
- State verification

**Tools:**
- tuistory skill for automation
- Terminal with sufficient size (80x24 minimum)

**Required environment:**
- Terminal with TTY support
- Zig and Bun installed
- Proper terminal environment (TERM set)

---

## Validation Concurrency

### Resource Cost Classification

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

### Isolation Strategy

**CLI validators:**
- Can run in parallel (no shared state beyond cache)
- Use different cache directories or coordinate to avoid conflicts

**API validators:**
- Must use different ports per validator
- Servers must be stopped after each test
- Models can be shared if tests don't modify

**TUI validators:**
- One TUI instance at a time (terminal exclusive)
- Use tuistory for isolation
- Clean state between runs

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

### Hardware Constraints

- GPU tests limited by hardware availability
- CPU fallback always available
- Some features (TurboQuant Triton) require GPU

**Mitigations:**
- Mark GPU-only tests
- Skip gracefully when GPU unavailable
- Test core algorithms on CPU

---

## Known Testing Gotchas

1. **Port conflicts:** Server tests must use unique ports
2. **Cache pollution:** Tests should clean up or use isolated cache dirs
3. **Process cleanup:** Server processes must be killed after tests
4. **Terminal state:** TUI tests need proper TTY setup
5. **Model size:** Large models cause timeouts - use small ones for tests
6. **HF rate limits:** Frequent downloads may trigger rate limiting

---

## Test Model Recommendations

For fast, reliable tests:

| Model | Size | Use Case |
|-------|------|----------|
| microsoft/DialoGPT-small | 117M | Download, basic inference |
| sshleifer/tiny-gpt2 | 17M | Very fast tests |
| distilbert-base-uncased | 66M | Classification tests |

For integration tests (if time permits):
| meta-llama/Llama-2-7b-hf | 7B | Real-world quantization |

---

## Validation Checklist

Before running validators:
- [ ] Python environment set up
- [ ] Package installed (pip install -e .)
- [ ] HF_TOKEN set (for gated models)
- [ ] Ports 3200-3299 available
- [ ] Cache directory writeable
- [ ] Network access to HuggingFace
- [ ] GPU available (optional)
