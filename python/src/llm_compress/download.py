"""Model download functionality for llm-compress.

This module provides functionality for downloading models from HuggingFace Hub
with progress bars, custom cache directories, and metadata management.
"""

import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


# Default cache directory for models
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "llm-compress"


class DownloadError(Exception):
    """Error during model download."""
    pass


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Get the cache directory for model downloads.
    
    Args:
        cache_dir: Custom cache directory. If None, uses default.
        
    Returns:
        Path to the cache directory.
    """
    if cache_dir:
        return Path(cache_dir)
    return Path(os.environ.get("LLM_COMPRESS_CACHE_DIR", DEFAULT_CACHE_DIR))


def _get_model_dir(cache_dir: Path, model_id: str) -> Path:
    """Get the directory for a specific model.
    
    Args:
        cache_dir: The root cache directory.
        model_id: The HuggingFace model identifier.
        
    Returns:
        Path to the model's directory.
    """
    # Replace / with -- to make valid directory name
    safe_model_id = model_id.replace("/", "--")
    return cache_dir / safe_model_id


def get_model_dir(model_id: str, cache_dir: str | Path | None = None) -> Path:
    """Get the directory for a specific model (public API).
    
    Args:
        model_id: The HuggingFace model identifier.
        cache_dir: Custom cache directory. If None, uses default.
        
    Returns:
        Path to the model's directory.
    """
    cache_path = get_cache_dir(cache_dir)
    return _get_model_dir(cache_path, model_id)


def _save_metadata(model_dir: Path, model_id: str, info: dict[str, Any]) -> None:
    """Save model metadata to the cache directory.
    
    Args:
        model_dir: The model's cache directory.
        model_id: The HuggingFace model identifier.
        info: Additional model information to save.
    """
    metadata = {
        "model_id": model_id,
        **info,
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_metadata(model_id: str, info: dict[str, Any], cache_dir: str | Path | None = None) -> None:
    """Save model metadata to the cache directory (public API).
    
    Args:
        model_id: The HuggingFace model identifier.
        info: Model information to save.
        cache_dir: Custom cache directory. If None, uses default.
    """
    cache_path = get_cache_dir(cache_dir)
    model_dir = _get_model_dir(cache_path, model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    _save_metadata(model_dir, model_id, info)


def _load_metadata(model_dir: Path) -> dict[str, Any] | None:
    """Load model metadata from the cache directory.
    
    Args:
        model_dir: The model's cache directory.
        
    Returns:
        Metadata dict if found, None otherwise.
    """
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def load_metadata(model_id: str, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Load model metadata from the cache directory (public API).
    
    Args:
        model_id: The HuggingFace model identifier.
        cache_dir: Custom cache directory. If None, uses default.
        
    Returns:
        Metadata dict if found, empty dict otherwise.
    """
    cache_path = get_cache_dir(cache_dir)
    model_dir = _get_model_dir(cache_path, model_id)
    metadata = _load_metadata(model_dir)
    return metadata if metadata is not None else {}


def download_model(
    model_id: str,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    """Download a model from HuggingFace Hub.
    
    This function downloads a model and its configuration files from the
    HuggingFace Hub, displays a progress bar during download, and saves
    metadata alongside the files.
    
    Args:
        model_id: The HuggingFace model identifier (e.g., microsoft/DialoGPT-medium).
        cache_dir: Custom cache directory. If None, uses default ~/.cache/llm-compress/.
        token: HuggingFace token for gated models. If None, uses HF_TOKEN env var.
        
    Returns:
        Path to the downloaded model directory.
        
    Raises:
        DownloadError: If the model doesn't exist or download fails.
        
    Example:
        >>> model_path = download_model("microsoft/DialoGPT-medium")
        >>> print(model_path)
        PosixPath('/home/user/.cache/llm-compress/microsoft--DialoGPT-medium')
    """
    cache_path = get_cache_dir(cache_dir)
    model_dir = _get_model_dir(cache_path, model_id)
    
    # Check if model already exists
    if model_dir.exists():
        metadata = _load_metadata(model_dir)
        if metadata and metadata.get("model_id") == model_id:
            print(f"Model {model_id} already exists in cache at {model_dir}")
            return model_dir
    
    # Verify model exists on HuggingFace before downloading
    api = HfApi(token=token)
    try:
        api.model_info(model_id)
    except RepositoryNotFoundError:
        raise DownloadError(f"Model '{model_id}' not found on HuggingFace Hub. "
                            "Please check the model ID and ensure it's correct.")
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            raise DownloadError(f"Model '{model_id}' requires authentication. "
                                "Please provide a valid HuggingFace token.")
        raise DownloadError(f"Error accessing HuggingFace Hub: {e}")
    
    print(f"Downloading {model_id} from HuggingFace Hub...")
    
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use snapshot_download - it has built-in progress bars
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        
        # Save metadata
        _save_metadata(
            model_dir,
            model_id,
            {
                "download_path": downloaded_path,
                "files_downloaded": len(list(model_dir.iterdir())),
            }
        )
        
        print(f"Download complete: {model_dir}")
        return model_dir
        
    except Exception as e:
        # Clean up partial downloads
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)
        raise DownloadError(f"Failed to download model '{model_id}': {e}")


def is_model_cached(model_id: str, cache_dir: str | Path | None = None) -> bool:
    """Check if a model is already cached locally.
    
    Args:
        model_id: The HuggingFace model identifier.
        cache_dir: Custom cache directory. If None, uses default.
        
    Returns:
        True if the model exists in cache with valid metadata.
    """
    cache_path = get_cache_dir(cache_dir)
    model_dir = _get_model_dir(cache_path, model_id)
    
    if not model_dir.exists():
        return False
    
    metadata = _load_metadata(model_dir)
    return metadata is not None and metadata.get("model_id") == model_id


def remove_cached_model(model_id: str, cache_dir: str | Path | None = None) -> None:
    """Remove a model from the cache.
    
    This function deletes all files associated with a model from the cache,
    including the downloaded model files and metadata.
    
    Args:
        model_id: The HuggingFace model identifier (e.g., microsoft/DialoGPT-medium).
        cache_dir: Custom cache directory. If None, uses default.
        
    Raises:
        DownloadError: If the model doesn't exist in cache or removal fails.
        
    Example:
        >>> remove_cached_model("microsoft/DialoGPT-medium")
        >>> # Model files have been deleted
    """
    cache_path = get_cache_dir(cache_dir)
    model_dir = _get_model_dir(cache_path, model_id)
    
    # Verify model exists in cache
    if not model_dir.exists():
        raise DownloadError(f"Model '{model_id}' not found in cache at {model_dir}")
    
    metadata = _load_metadata(model_dir)
    if metadata is None or metadata.get("model_id") != model_id:
        raise DownloadError(f"Model '{model_id}' not found in cache (metadata missing or invalid)")
    
    try:
        import shutil
        shutil.rmtree(model_dir)
    except Exception as e:
        raise DownloadError(f"Failed to remove model '{model_id}': {e}")


def list_cached_models(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List all models currently in the cache.
    
    Args:
        cache_dir: Custom cache directory. If None, uses default.
        
    Returns:
        List of model metadata dictionaries.
    """
    cache_path = get_cache_dir(cache_dir)
    models = []
    
    if not cache_path.exists():
        return models
    
    for model_dir in cache_path.iterdir():
        if model_dir.is_dir():
            metadata = _load_metadata(model_dir)
            if metadata:
                metadata["local_path"] = str(model_dir)
                models.append(metadata)
    
    return models
