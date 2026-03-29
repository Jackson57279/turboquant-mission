#!/bin/bash
# llm-compress environment setup script
# This script is idempotent - safe to run multiple times

set -e

echo "Setting up llm-compress development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python 3.10+
required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.10+ required, found $python_version"
    exit 1
fi

# Install Python dependencies if pyproject.toml exists
if [ -f "python/pyproject.toml" ]; then
    echo "Installing Python package in editable mode..."
    cd python
    pip install -e ".[dev]" --quiet 2>/dev/null || pip install -e ".[dev]"
    cd ..
fi

# Check for Bun (required for TypeScript/OpenTUI)
if ! command -v bun &> /dev/null; then
    echo "Bun not found. Installing Bun..."
    curl -fsSL https://bun.sh/install | bash
    export PATH="$HOME/.bun/bin:$PATH"
fi

# Check for Zig (required for OpenTUI)
if ! command -v zig &> /dev/null; then
    echo "Zig not found. Please install Zig from https://ziglang.org/download/"
    echo "For most systems: curl -fsSL https://ziglang.org/download/0.15.2/zig-linux-x86_64-0.15.2.tar.xz | tar -xJf - && sudo mv zig-linux-x86_64-0.15.2/zig /usr/local/bin/"
fi

# Install TypeScript dependencies if package.json exists
if [ -f "typescript/package.json" ]; then
    echo "Installing TypeScript dependencies..."
    cd typescript
    bun install
    cd ..
fi

# Create cache directory
mkdir -p ~/.cache/llm-compress

# Check for CUDA (optional)
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "CUDA found: $cuda_version"
else
    echo "CUDA not found. GPU acceleration will not be available."
fi

# Check for HuggingFace token (optional)
if [ -z "$HF_TOKEN" ]; then
    echo "Note: HF_TOKEN not set. Some gated models may not be accessible."
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run CLI: llm-compress --help"
echo "  2. Run TUI: llm-compress tui"
echo "  3. Run tests: pytest python/tests/"
