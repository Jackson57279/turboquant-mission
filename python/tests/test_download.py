"""Tests for the download module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_compress.cli import main
from llm_compress.download import (
    DEFAULT_CACHE_DIR,
    DownloadError,
    _get_model_dir,
    _load_metadata,
    _save_metadata,
    download_model,
    get_cache_dir,
    is_model_cached,
    list_cached_models,
)


class TestGetCacheDir:
    """Tests for get_cache_dir function."""

    def test_default_cache_dir(self):
        """Test default cache directory is ~/.cache/llm-compress."""
        result = get_cache_dir()
        assert result == DEFAULT_CACHE_DIR
        assert result.name == "llm-compress"

    def test_custom_cache_dir(self):
        """Test custom cache directory is used when provided."""
        custom_dir = "/tmp/test-cache"
        result = get_cache_dir(custom_dir)
        assert result == Path(custom_dir)

    def test_env_var_cache_dir(self):
        """Test environment variable overrides default."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_dir = os.path.join(tmp_dir, "env-cache")
            with patch.dict(os.environ, {"LLM_COMPRESS_CACHE_DIR": env_dir}):
                result = get_cache_dir()
                assert result == Path(env_dir)


class TestGetModelDir:
    """Tests for _get_model_dir function."""

    def test_model_dir_naming(self):
        """Test model directory replaces / with --."""
        cache_dir = Path("/cache")
        model_dir = _get_model_dir(cache_dir, "microsoft/DialoGPT-medium")
        assert model_dir == Path("/cache/microsoft--DialoGPT-medium")

    def test_model_dir_no_slash(self):
        """Test model directory without slash."""
        cache_dir = Path("/cache")
        model_dir = _get_model_dir(cache_dir, "gpt2")
        assert model_dir == Path("/cache/gpt2")


class TestMetadata:
    """Tests for metadata save/load functions."""

    def test_save_and_load_metadata(self, tmp_path):
        """Test saving and loading metadata."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        info = {"key": "value", "number": 42}
        _save_metadata(model_dir, "test/model", info)

        loaded = _load_metadata(model_dir)
        assert loaded is not None
        assert loaded["model_id"] == "test/model"
        assert loaded["key"] == "value"
        assert loaded["number"] == 42

    def test_load_missing_metadata(self, tmp_path):
        """Test loading metadata from directory without metadata file."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        loaded = _load_metadata(model_dir)
        assert loaded is None

    def test_metadata_json_format(self, tmp_path):
        """Test metadata is saved as valid JSON."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        _save_metadata(model_dir, "test/model", {"test": "data"})

        metadata_path = model_dir / "metadata.json"
        assert metadata_path.exists()

        # Verify it's valid JSON
        with open(metadata_path) as f:
            data = json.load(f)
        assert data["model_id"] == "test/model"


class TestDownloadModel:
    """Tests for download_model function."""

    @patch("llm_compress.download.HfApi")
    @patch("llm_compress.download.snapshot_download")
    def test_successful_download(self, mock_snapshot, mock_hfapi, tmp_path):
        """Test successful model download."""
        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api

        model_dir = tmp_path / "models"
        mock_snapshot.return_value = str(model_dir / "microsoft--DialoGPT-medium")

        result = download_model(
            "microsoft/DialoGPT-medium",
            cache_dir=str(model_dir),
        )

        assert mock_api.model_info.called
        assert mock_snapshot.called
        assert result.exists()

    @patch("llm_compress.download.HfApi")
    def test_model_not_found(self, mock_hfapi, tmp_path):
        """Test error handling for non-existent model."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_api = MagicMock()
        mock_api.model_info.side_effect = RepositoryNotFoundError(
            "Model not found", response=mock_response
        )
        mock_hfapi.return_value = mock_api

        with pytest.raises(DownloadError) as exc_info:
            download_model("invalid/model", cache_dir=str(tmp_path))

        assert "not found" in str(exc_info.value).lower()

    @patch("llm_compress.download.HfApi")
    def test_authentication_error(self, mock_hfapi, tmp_path):
        """Test error handling for gated models."""
        from huggingface_hub.utils import HfHubHTTPError

        mock_response = MagicMock()
        mock_response.status_code = 401

        mock_api = MagicMock()
        mock_api.model_info.side_effect = HfHubHTTPError(
            "Unauthorized", response=mock_response
        )
        mock_hfapi.return_value = mock_api

        with pytest.raises(DownloadError) as exc_info:
            download_model("private/model", cache_dir=str(tmp_path))

        assert "authentication" in str(exc_info.value).lower() or "token" in str(exc_info.value).lower()

    @patch("llm_compress.download.HfApi")
    @patch("llm_compress.download.snapshot_download")
    @patch("llm_compress.download._load_metadata")
    def test_already_cached(self, mock_load_meta, mock_snapshot, mock_hfapi, tmp_path):
        """Test that already cached models are detected."""
        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api
        mock_load_meta.return_value = {"model_id": "microsoft/DialoGPT-medium"}

        model_dir = tmp_path / "models"
        model_path = model_dir / "microsoft--DialoGPT-medium"
        model_path.mkdir(parents=True)

        result = download_model(
            "microsoft/DialoGPT-medium",
            cache_dir=str(model_dir),
        )

        # Should not call model_info or snapshot_download for cached models
        assert not mock_api.model_info.called
        assert not mock_snapshot.called
        assert result == model_path


class TestIsModelCached:
    """Tests for is_model_cached function."""

    def test_model_cached(self, tmp_path):
        """Test detecting cached model."""
        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        model_dir.mkdir()

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({"model_id": "microsoft/DialoGPT-medium"}, f)

        assert is_model_cached("microsoft/DialoGPT-medium", cache_dir=str(tmp_path))

    def test_model_not_cached(self, tmp_path):
        """Test detecting non-cached model."""
        assert not is_model_cached("microsoft/DialoGPT-medium", cache_dir=str(tmp_path))

    def test_cache_dir_missing(self, tmp_path):
        """Test when cache directory doesn't exist."""
        non_existent = tmp_path / "does-not-exist"
        assert not is_model_cached("any/model", cache_dir=str(non_existent))


class TestListCachedModels:
    """Tests for list_cached_models function."""

    def test_empty_cache(self, tmp_path):
        """Test listing empty cache."""
        result = list_cached_models(cache_dir=str(tmp_path))
        assert result == []

    def test_list_cached_models(self, tmp_path):
        """Test listing cached models."""
        # Create two model directories with metadata
        model1 = tmp_path / "model1"
        model1.mkdir()
        with open(model1 / "metadata.json", "w") as f:
            json.dump({"model_id": "org/model1"}, f)

        model2 = tmp_path / "model2"
        model2.mkdir()
        with open(model2 / "metadata.json", "w") as f:
            json.dump({"model_id": "org/model2"}, f)

        result = list_cached_models(cache_dir=str(tmp_path))

        assert len(result) == 2
        model_ids = {r["model_id"] for r in result}
        assert model_ids == {"org/model1", "org/model2"}


class TestDownloadCLI:
    """Tests for download CLI command."""

    @patch("llm_compress.download.snapshot_download")
    @patch("llm_compress.download.HfApi")
    def test_download_success(self, mock_hfapi, mock_snapshot, tmp_path):
        """Test successful download command."""
        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api

        model_dir = tmp_path / "models"
        mock_snapshot.return_value = str(model_dir / "microsoft--DialoGPT-medium")

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["download", "microsoft/DialoGPT-medium"])

        assert result.exit_code == 0

    @patch("llm_compress.download.snapshot_download")
    @patch("llm_compress.download.HfApi")
    def test_download_with_cache_dir(self, mock_hfapi, mock_snapshot, tmp_path):
        """Test download with custom cache directory."""
        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api

        cache_dir = tmp_path / "custom-cache"
        mock_snapshot.return_value = str(cache_dir / "microsoft--DialoGPT-medium")

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, [
                "download",
                "microsoft/DialoGPT-medium",
                "--cache-dir", str(cache_dir),
            ])

        assert result.exit_code == 0
        # Verify snapshot_download was called with correct cache_dir
        call_kwargs = mock_snapshot.call_args.kwargs
        assert "local_dir" in call_kwargs

    @patch("llm_compress.download.HfApi")
    def test_download_invalid_model(self, mock_hfapi, tmp_path):
        """Test download with invalid model ID."""
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_api = MagicMock()
        mock_api.model_info.side_effect = RepositoryNotFoundError(
            "Model not found", response=mock_response
        )
        mock_hfapi.return_value = mock_api

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["download", "invalid/model"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    @patch("llm_compress.download.snapshot_download")
    @patch("llm_compress.download.HfApi")
    def test_download_with_token(self, mock_hfapi, mock_snapshot, tmp_path):
        """Test download with token option."""
        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api

        mock_snapshot.return_value = str(tmp_path / "private--model")

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, [
                "download",
                "private/model",
                "--token", "hf_test_token_123",
            ])

        assert result.exit_code == 0

    def test_download_help(self):
        """Test download command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--help"])

        assert result.exit_code == 0
        assert "Download a model" in result.output
        assert "--cache-dir" in result.output
        assert "--token" in result.output
        assert "MODEL_ID" in result.output
