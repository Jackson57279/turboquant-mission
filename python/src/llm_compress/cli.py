"""LLM Compress CLI entry point.

This module provides the command-line interface for llm-compress,
including commands for downloading, quantizing, and serving LLMs.
"""

import click
from llm_compress import __version__


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Human-readable size string (e.g., "1.5 MB").
    """
    if size_bytes == 0:
        return "0 B"
    
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


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
    from llm_compress.download import download_model, DownloadError
    
    try:
        model_path = download_model(
            model_id=model_id,
            cache_dir=cache_dir,
            token=token,
        )
        click.echo(f"Model downloaded to: {model_path}")
    except DownloadError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.ClickException(str(e))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.ClickException(f"Download failed: {e}")


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
    """List all downloaded models.
    
    Displays a table showing model IDs, sizes, and download dates.
    Shows a friendly message if the cache is empty.
    """
    from llm_compress.download import list_cached_models, get_cache_dir
    from pathlib import Path
    
    cache_path = get_cache_dir(cache_dir)
    models = list_cached_models(cache_dir=cache_path)
    
    if not models:
        click.echo("No models found in cache.")
        click.echo(f"Cache directory: {cache_path}")
        click.echo("Use 'llm-compress download <model_id>' to download a model.")
        return
    
    # Calculate column widths
    max_id_len = max(len(m.get("model_id", "")) for m in models)
    max_id_len = max(max_id_len, len("MODEL ID"))
    
    # Header
    click.echo(f"{'MODEL ID':<{max_id_len}}  {'SIZE':>10}  {'FILES':>6}  {'DOWNLOADED':>12}")
    click.echo("-" * (max_id_len + 10 + 6 + 12 + 6))
    
    # Rows
    for model in models:
        model_id = model.get("model_id", "unknown")
        local_path = model.get("local_path")
        files_downloaded = model.get("files_downloaded", 0)
        
        # Calculate size
        total_size = 0
        if local_path:
            path = Path(local_path)
            if path.exists():
                for item in path.rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
        
        size_str = _format_size(total_size)
        click.echo(f"{model_id:<{max_id_len}}  {size_str:>10}  {files_downloaded:>6}  {'--':>12}")
    
    click.echo()
    click.echo(f"Cache directory: {cache_path}")
    click.echo(f"Total models: {len(models)}")


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
    
    This command deletes all files associated with the model from the cache,
    including the downloaded model files and metadata.
    
    Examples:
        llm-compress remove microsoft/DialoGPT-medium
        llm-compress remove microsoft/DialoGPT-medium --force
    """
    from llm_compress.download import remove_cached_model, DownloadError, is_model_cached
    
    # Check if model exists before confirming
    if not is_model_cached(model_id, cache_dir=cache_dir):
        raise click.ClickException(f"Model '{model_id}' not found in cache.")
    
    if not force:
        click.confirm(f"Remove '{model_id}' from cache?", abort=True)
    
    try:
        remove_cached_model(model_id, cache_dir=cache_dir)
        click.echo(f"Model '{model_id}' removed successfully.")
    except DownloadError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Failed to remove model: {e}")


@main.command()
def tui() -> None:
    """Launch the terminal user interface."""
    click.echo("Launching TUI...")
    click.echo("Note: This is a placeholder. Full implementation coming in tui-main-interface.")


if __name__ == "__main__":
    main()
