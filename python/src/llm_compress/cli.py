"""LLM Compress CLI entry point.

This module provides the command-line interface for llm-compress,
including commands for downloading, quantizing, and serving LLMs.
"""

import logging
import click

from llm_compress import __version__

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    from llm_compress.download import DownloadError, download_model

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
    help="Enable KV cache quantization (3-bit keys, 2-bit values)",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Cache directory for model files",
)
def quantize(model_id: str, bits: str, kv_cache: bool, cache_dir: str | None) -> None:
    """Quantize a downloaded model.
    
    MODEL_ID is the HuggingFace model identifier.
    
    This command quantizes a previously downloaded model's weights to the
    specified bit width (4 or 8). The quantized model is saved alongside
    the original and can be used for efficient inference.
    
    Examples:
        llm-compress quantize microsoft/DialoGPT-medium --bits 4
        llm-compress quantize meta-llama/Llama-2-7b-hf --bits 8 --kv-cache
    """
    import time

    from llm_compress.download import is_model_cached
    from llm_compress.quantization.weight import get_compression_ratio, quantize_model

    # Check if model is downloaded
    if not is_model_cached(model_id, cache_dir=cache_dir):
        raise click.ClickException(
            f"Model '{model_id}' not found in cache. "
            f"Please download it first with: llm-compress download {model_id}"
        )

    bits_int = int(bits)

    click.echo(f"Quantizing {model_id} to {bits}-bit weights...")

    if kv_cache:
        click.echo("KV cache quantization enabled (3-bit keys, 2-bit values)")

    # Show progress
    click.echo("Loading model and preparing quantization...")
    start_time = time.time()

    try:
        # Perform quantization
        output_dir = quantize_model(
            model_id=model_id,
            bits=bits_int,
            cache_dir=cache_dir,
        )

        # Calculate compression ratio
        compression_ratio = get_compression_ratio(output_dir)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Output results
        click.echo()
        click.echo("✓ Quantization complete!")
        click.echo(f"  Model: {model_id}")
        click.echo(f"  Bits: {bits}-bit weights")
        click.echo(f"  KV cache: {'enabled' if kv_cache else 'disabled'}")
        click.echo(f"  Compression ratio: {compression_ratio:.2f}x")
        click.echo(f"  Time: {elapsed_time:.1f}s")
        click.echo(f"  Output: {output_dir}")
        click.echo()
        click.echo("To serve this model, run:")
        click.echo(f"  llm-compress serve {model_id}")

    except ValueError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Quantization failed: {e}")


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
@click.option(
    "--kv-cache/--no-kv-cache",
    default=True,
    help="Enable KV cache compression (default: enabled)",
)
def serve(model_id: str, port: int, host: str, backend: str, cache_dir: str | None, kv_cache: bool) -> None:
    """Start the OpenAI-compatible API server.
    
    MODEL_ID is the HuggingFace model identifier of the quantized model.
    
    This command starts a FastAPI server that provides an OpenAI-compatible
    API for serving quantized models. The server supports:
    
    - GET /health - Health check endpoint
    - GET /v1/models - List available models
    - POST /v1/chat/completions - Chat completions (with streaming support)
    - POST /v1/completions - Legacy text completions (with streaming support)
    
    Examples:
        llm-compress serve microsoft/DialoGPT-medium
        llm-compress serve meta-llama/Llama-2-7b-hf --backend llama-cpp --port 8080
        llm-compress serve my-model --host 0.0.0.0 --port 3200
    """
    import uvicorn
    
    from llm_compress.download import is_model_cached, get_cache_dir, load_metadata
    from llm_compress.server.app import create_app
    from pathlib import Path

    # Check if model is downloaded
    is_cached = is_model_cached(model_id, cache_dir=cache_dir)
    
    if not is_cached:
        # Check if it's in default cache location
        click.echo(f"Warning: Model '{model_id}' not found in cache.", err=True)
        click.echo("Attempting to load from HuggingFace Hub...", err=True)
    else:
        # Check for quantization by looking for quantization metadata
        model_cache_dir = get_cache_dir(cache_dir)
        model_dir = model_cache_dir / model_id.replace("/", "--")
        
        # Check if quantized version exists
        quantized_4bit_dir = model_dir / "quantized-4bit"
        quantized_8bit_dir = model_dir / "quantized-8bit"
        
        is_quantized = quantized_4bit_dir.exists() or quantized_8bit_dir.exists()
        
        # Also check metadata
        metadata = load_metadata(model_id, cache_dir=cache_dir)
        is_quantized = is_quantized or metadata.get("quantized", False)
        
        if not is_quantized:
            click.echo(
                f"Warning: Model '{model_id}' appears to be unquantized. "
                f"Performance may be reduced. Consider quantizing first:\n"
                f"  llm-compress quantize {model_id} --bits 4",
                err=True
            )
        else:
            quantization_info = metadata.get("quantization", {})
            bits = quantization_info.get("bits", "4")
            click.echo(f"Serving quantized model: {model_id} ({bits}-bit)")

    click.echo(f"Backend: {backend}")
    click.echo(f"Host: {host}")
    click.echo(f"Port: {port}")
    click.echo(f"KV cache compression: {'enabled' if kv_cache else 'disabled'}")
    click.echo()

    # Create the FastAPI app
    try:
        logger.info(f"Initializing {backend} backend with model {model_id}")
        app = create_app(
            model_id=model_id,
            backend=backend,
            cache_dir=cache_dir,
            enable_kv_compression=kv_cache,
        )
        logger.info("FastAPI application created successfully")
    except Exception as e:
        logger.error(f"Failed to create server: {e}")
        raise click.ClickException(f"Failed to create server: {e}")

    click.echo(f"Starting API server at http://{host}:{port}")
    click.echo("Available endpoints:")
    click.echo(f"  GET  http://{host}:{port}/health")
    click.echo(f"  GET  http://{host}:{port}/v1/models")
    click.echo(f"  POST http://{host}:{port}/v1/chat/completions")
    click.echo(f"  POST http://{host}:{port}/v1/completions")
    click.echo()
    click.echo("Press Ctrl+C to stop the server")
    click.echo()

    # Run the server
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        click.echo("\nServer stopped.")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise click.ClickException(f"Server error: {e}")


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
    from pathlib import Path

    from llm_compress.download import get_cache_dir, list_cached_models

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
    from llm_compress.download import DownloadError, is_model_cached, remove_cached_model

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
