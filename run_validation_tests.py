#!/usr/bin/env python3
"""User testing validation for core-python milestone CLI assertions."""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add python src to path
sys.path.insert(0, '/home/dih/turboquant-mission/python/src')

from llm_compress.cli import main
from llm_compress import __version__
from click.testing import CliRunner

runner = CliRunner()

results = {
    "milestone": "core-python",
    "timestamp": None,
    "assertions": {}
}

print("=" * 60)
print("User Testing Validation: core-python CLI Assertions")
print("=" * 60)

# VAL-CLI-014: Version command
print("\n--- VAL-CLI-014: Version command ---")
result = runner.invoke(main, ['--version'])
version_passed = result.exit_code == 0 and '0.1.0' in result.output
results["assertions"]["VAL-CLI-014"] = {
    "status": "passed" if version_passed else "failed",
    "exit_code": result.exit_code,
    "output_preview": result.output[:200]
}
print(f"exit_code: {result.exit_code}")
print(f"output: {result.output.strip()}")
print(f"VAL-CLI-014: {'PASSED' if version_passed else 'FAILED'}")

# VAL-CLI-013: Help command
print("\n--- VAL-CLI-013: Help command ---")
result = runner.invoke(main, ['--help'])
has_commands = all(cmd in result.output for cmd in ['download', 'quantize', 'serve', 'list', 'remove', 'tui'])
help_passed = result.exit_code == 0 and has_commands
results["assertions"]["VAL-CLI-013"] = {
    "status": "passed" if help_passed else "failed",
    "exit_code": result.exit_code,
    "has_all_commands": has_commands
}
print(f"exit_code: {result.exit_code}")
print(f"has_all_commands: {has_commands}")
print(f"VAL-CLI-013: {'PASSED' if help_passed else 'FAILED'}")

# Test subcommand helps
print("\n--- Subcommand help tests ---")
for cmd in ['download', 'quantize', 'serve', 'list', 'remove', 'tui']:
    result = runner.invoke(main, [cmd, '--help'])
    passed = result.exit_code == 0 and 'Usage:' in result.output
    print(f"  {cmd} --help: {'PASSED' if passed else 'FAILED'}")

# VAL-CLI-011: List models (empty cache)
print("\n--- VAL-CLI-011: List models (empty) ---")
with tempfile.TemporaryDirectory() as tmpdir:
    cache_dir = os.path.join(tmpdir, 'cache')
    result = runner.invoke(main, ['list', '--cache-dir', cache_dir])
    list_passed = result.exit_code == 0 and 'No models found' in result.output
    results["assertions"]["VAL-CLI-011-empty"] = {
        "status": "passed" if list_passed else "failed",
        "exit_code": result.exit_code
    }
    print(f"exit_code: {result.exit_code}")
    print(f"output: {result.output.strip()[:200]}")
    print(f"VAL-CLI-011 (empty): {'PASSED' if list_passed else 'FAILED'}")

# VAL-CLI-003: Invalid model ID error
print("\n--- VAL-CLI-003: Invalid model ID ---")
with tempfile.TemporaryDirectory() as tmpdir:
    cache_dir = os.path.join(tmpdir, 'cache')
    result = runner.invoke(main, ['download', 'invalid/model/123', '--cache-dir', cache_dir])
    # Should fail with non-zero exit code
    invalid_passed = result.exit_code != 0
    results["assertions"]["VAL-CLI-003"] = {
        "status": "passed" if invalid_passed else "failed",
        "exit_code": result.exit_code
    }
    print(f"exit_code: {result.exit_code}")
    print(f"VAL-CLI-003: {'PASSED' if invalid_passed else 'FAILED'}")

# VAL-CLI-001 and VAL-CLI-002: Download model
print("\n--- VAL-CLI-001 & VAL-CLI-002: Download model ---")
with tempfile.TemporaryDirectory() as tmpdir:
    cache_dir = os.path.join(tmpdir, 'cache')
    print(f"Downloading to: {cache_dir}")
    print("This may take 1-2 minutes...")
    result = runner.invoke(main, ['download', 'sshleifer/tiny-gpt2', '--cache-dir', cache_dir], timeout=300)
    
    # Check if download succeeded
    model_dir = os.path.join(cache_dir, 'sshleifer--tiny-gpt2')
    model_exists = os.path.exists(model_dir)
    
    download_passed = result.exit_code == 0 and model_exists
    results["assertions"]["VAL-CLI-001"] = {
        "status": "passed" if download_passed else "failed",
        "exit_code": result.exit_code,
        "model_exists": model_exists
    }
    print(f"exit_code: {result.exit_code}")
    print(f"model_exists: {model_exists}")
    print(f"VAL-CLI-001: {'PASSED' if download_passed else 'FAILED'}")
    
    # VAL-CLI-002: Custom cache directory - model is in the specified dir
    cache_dir_passed = model_exists
    results["assertions"]["VAL-CLI-002"] = {
        "status": "passed" if cache_dir_passed else "failed",
        "cache_dir_used": cache_dir,
        "model_path": model_dir if model_exists else None
    }
    print(f"VAL-CLI-002: {'PASSED' if cache_dir_passed else 'FAILED'}")
    
    # VAL-CLI-011: List with model present
    print("\n--- VAL-CLI-011: List models (with model) ---")
    result = runner.invoke(main, ['list', '--cache-dir', cache_dir])
    list_with_model = result.exit_code == 0 and 'sshleifer/tiny-gpt2' in result.output
    results["assertions"]["VAL-CLI-011-with-model"] = {
        "status": "passed" if list_with_model else "failed",
        "exit_code": result.exit_code,
        "model_in_list": 'sshleifer/tiny-gpt2' in result.output
    }
    print(f"exit_code: {result.exit_code}")
    print(f"model_in_list: {'sshleifer/tiny-gpt2' in result.output}")
    print(f"VAL-CLI-011 (with model): {'PASSED' if list_with_model else 'FAILED'}")
    
    # VAL-CLI-012: Remove model
    print("\n--- VAL-CLI-012: Remove model ---")
    result = runner.invoke(main, ['remove', 'sshleifer/tiny-gpt2', '--cache-dir', cache_dir, '--force'])
    model_removed = not os.path.exists(model_dir)
    remove_passed = result.exit_code == 0 and model_removed
    results["assertions"]["VAL-CLI-012"] = {
        "status": "passed" if remove_passed else "failed",
        "exit_code": result.exit_code,
        "model_removed": model_removed
    }
    print(f"exit_code: {result.exit_code}")
    print(f"model_removed: {model_removed}")
    print(f"VAL-CLI-012: {'PASSED' if remove_passed else 'FAILED'}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for a in results["assertions"].values() if a["status"] == "passed")
failed = sum(1 for a in results["assertions"].values() if a["status"] == "failed")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

# Save results
output_dir = Path('/home/dih/turboquant-mission/.factory/validation/core-python/user-testing/flows')
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'cli-all.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_dir / 'cli-all.json'}")
