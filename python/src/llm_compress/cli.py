"""LLM Compress CLI entry point.

This module provides the command-line interface for llm-compress,
including commands for downloading, quantizing, and serving LLMs.
"""

import click
from llm_compress import __version__


@click.group()
@click.version_option(version=__version__, prog_name="llm-compress")
def main() -> None:
    """LLM Compress: Unified LLM Quantization & Inference System.
    
    This CLI provides commands for:
    - download: Download models from HuggingFace Hub
    - quantize: Quantize models to 4-bit or 8-bit weights
    - serve: Start OpenAI-compatible API server
    - list: List downloaded models
    - remove: Remove a downloaded model
    - tui: Launch the terminal user interface
    
    Examples:
        llm-compress download meta-llama/Llama-2-7b-hf
        llm-compress quantize meta-llama/Llama-2-7b-hf --bits 4
        llm-compress serve meta-llama/Llama-2-7b-hf --port 3200
    """
    pass


@main.command()
@click.argument("model_id")
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Custom cache directory for model files",
)
@click.option(
    "--token",
    envvar="HF_TOKEN",
    help="HuggingFace token for gated models",
)
def download(model_id: str, cache_dir: str | None, token: str | None) -> None:
    """Download a model from HuggingFace Hub.
    
    MODEL_ID is the HuggingFace model identifier (e.g., meta-llama/Llama-2-7b-hf).
    """
    click.echo(f"Downloading {model_id}...")
    click.echo("Note: This is a placeholder. Full implementation coming in cli-download-command.")


@main.command()
@click.argument("model_id")
@click.option(
    "--bits",
    type=click.Choice(["4", "8"]),
    default="4",
    help="Quantization bit width (4 or 8)",
)
@click.option(
    "--kv-cache/--no-kv-cache",
    default=False,
    help="Enable KV cache quantization",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Cache directory for model files",
)
def quantize(model_id: str, bits: str, kv_cache: bool, cache_dir: str | None) -> None:
    """Quantize a downloaded model.
    
    MODEL_ID is the HuggingFace model identifier.
    """
    click.echo(f"Quantizing {model_id} to {bits}-bit weights...")
    if kv_cache:
        click.echo("KV cache quantization enabled (3-bit keys, 2-bit values)")
    click.echo("Note: This is a placeholder. Full implementation coming in cli-quantize-command.")


@main.command()
@click.argument("model_id")
@click.option(
    "--port",
    type=int,
    default=3200,
    help="Server port (default: 3200)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Server host (default: 127.0.0.1)",
)
@click.option(
    "--backend",
    type=click.Choice(["vllm", "llama-cpp"]),
    default="vllm",
    help="Inference backend (default: vllm)",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Cache directory for model files",
)
def serve(model_id: str, port: int, host: str, backend: str, cache_dir: str | None) -> None:
    """Start the OpenAI-compatible API server.
    
    MODEL_ID is the HuggingFace model identifier of the quantized model.
    """
    click.echo(f"Starting API server for {model_id}...")
    click.echo(f"Backend: {backend}, Host: {host}, Port: {port}")
    click.echo("Note: This is a placeholder. Full implementation coming in cli-serve-command.")


@main.command()
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Cache directory for model files",
)
def list(cache_dir: str | None) -> None:
    """List all downloaded models."""
    click.echo("Cached models:")
    click.echo("Note: This is a placeholder. Full implementation coming in cli-list-remove-commands.")


@main.command()
@click.argument("model_id")
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Cache directory for model files",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Skip confirmation prompt",
)
def remove(model_id: str, cache_dir: str | None, force: bool) -> None:
    """Remove a downloaded model.
    
    MODEL_ID is the HuggingFace model identifier.
    """
    if not force:
        click.confirm(f"Remove {model_id}?", abort=True)
    click.echo(f"Removing {model_id}...")
    click.echo("Note: This is a placeholder. Full implementation coming in cli-list-remove-commands.")


@main.command()
def tui() -> None:
    """Launch the terminal user interface."""
    click.echo("Launching TUI...")
    click.echo("Note: This is a placeholder. Full implementation coming in tui-main-interface.")


if __name__ == "__main__":
    main()
