"""Tests for the list and remove CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_compress.cli import main
from llm_compress.download import (
    DownloadError,
    _get_model_dir,
    get_cache_dir,
    is_model_cached,
    list_cached_models,
    remove_cached_model,
)


class TestListCachedModels:
    """Tests for list_cached_models function."""

    def test_empty_cache(self, tmp_path):
        """Test listing empty cache."""
        result = list_cached_models(cache_dir=str(tmp_path))
        assert result == []

    def test_list_models_with_metadata(self, tmp_path):
        """Test listing models with valid metadata."""
        # Create two model directories with metadata
        model1 = tmp_path / "org--model1"
        model1.mkdir()
        with open(model1 / "metadata.json", "w") as f:
            json.dump({"model_id": "org/model1", "files_downloaded": 5}, f)

        model2 = tmp_path / "org--model2"
        model2.mkdir()
        with open(model2 / "metadata.json", "w") as f:
            json.dump({"model_id": "org/model2", "files_downloaded": 3}, f)

        result = list_cached_models(cache_dir=str(tmp_path))

        assert len(result) == 2
        model_ids = {r["model_id"] for r in result}
        assert model_ids == {"org/model1", "org/model2"}

    def test_list_ignores_invalid_directories(self, tmp_path):
        """Test that directories without metadata are ignored."""
        # Create valid model
        model1 = tmp_path / "org--model1"
        model1.mkdir()
        with open(model1 / "metadata.json", "w") as f:
            json.dump({"model_id": "org/model1"}, f)

        # Create invalid directory (no metadata)
        model2 = tmp_path / "org--model2"
        model2.mkdir()
        # Don't create metadata.json

        result = list_cached_models(cache_dir=str(tmp_path))

        assert len(result) == 1
        assert result[0]["model_id"] == "org/model1"


class TestRemoveCachedModel:
    """Tests for remove_cached_model function."""

    def test_successful_removal(self, tmp_path):
        """Test successfully removing a cached model."""
        # Create model directory with metadata
        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()
        (model_dir / "config.json").touch()
        with open(model_dir / "metadata.json", "w") as f:
            json.dump({"model_id": "microsoft/DialoGPT-medium", "files_downloaded": 2}, f)

        # Verify model exists
        assert model_dir.exists()

        # Remove the model
        remove_cached_model("microsoft/DialoGPT-medium", cache_dir=str(tmp_path))

        # Verify model is gone
        assert not model_dir.exists()

    def test_remove_nonexistent_model(self, tmp_path):
        """Test removing a model that doesn't exist."""
        with pytest.raises(DownloadError) as exc_info:
            remove_cached_model("nonexistent/model", cache_dir=str(tmp_path))

        assert "not found" in str(exc_info.value).lower()

    def test_remove_model_without_metadata(self, tmp_path):
        """Test removing a model directory without metadata file."""
        # Create model directory without metadata
        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        model_dir.mkdir()

        with pytest.raises(DownloadError) as exc_info:
            remove_cached_model("microsoft/DialoGPT-medium", cache_dir=str(tmp_path))

        assert "metadata" in str(exc_info.value).lower()


class TestListCLI:
    """Tests for list CLI command."""

    def test_list_empty_cache(self, tmp_path):
        """Test list command with empty cache."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["list", "--cache-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "No models found" in result.output
        assert "llm-compress download" in result.output

    def test_list_with_models(self, tmp_path):
        """Test list command with cached models."""
        # Create model directories with files
        model1 = tmp_path / "org--model1"
        model1.mkdir()
        (model1 / "model.bin").write_bytes(b"x" * 1024)  # 1KB
        with open(model1 / "metadata.json", "w") as f:
            json.dump({"model_id": "org/model1", "files_downloaded": 1}, f)

        model2 = tmp_path / "org--model2"
        model2.mkdir()
        (model2 / "model.bin").write_bytes(b"x" * 2048)  # 2KB
        with open(model2 / "metadata.json", "w") as f:
            json.dump({"model_id": "org/model2", "files_downloaded": 1}, f)

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--cache-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "MODEL ID" in result.output
        assert "SIZE" in result.output
        assert "org/model1" in result.output
        assert "org/model2" in result.output
        assert "1.0 KB" in result.output or "1024 B" in result.output
        assert "Total models: 2" in result.output

    def test_list_help(self):
        """Test list command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["list", "--help"])

        assert result.exit_code == 0
        assert "List all downloaded models" in result.output
        assert "--cache-dir" in result.output


class TestRemoveCLI:
    """Tests for remove CLI command."""

    def test_remove_with_confirmation(self, tmp_path):
        """Test remove command with confirmation prompt."""
        # Create model directory
        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()
        with open(model_dir / "metadata.json", "w") as f:
            json.dump({"model_id": "microsoft/DialoGPT-medium"}, f)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["remove", "microsoft/DialoGPT-medium", "--cache-dir", str(tmp_path)],
            input="y\n"  # Confirm removal
        )

        assert result.exit_code == 0
        assert "removed successfully" in result.output
        assert not model_dir.exists()

    def test_remove_cancelled(self, tmp_path):
        """Test remove command with cancelled confirmation."""
        # Create model directory
        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        model_dir.mkdir()
        with open(model_dir / "metadata.json", "w") as f:
            json.dump({"model_id": "microsoft/DialoGPT-medium"}, f)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["remove", "microsoft/DialoGPT-medium", "--cache-dir", str(tmp_path)],
            input="n\n"  # Cancel removal
        )

        assert result.exit_code != 0  # Should abort
        assert model_dir.exists()  # Model should still exist

    def test_remove_with_force(self, tmp_path):
        """Test remove command with --force flag."""
        # Create model directory
        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        model_dir.mkdir()
        with open(model_dir / "metadata.json", "w") as f:
            json.dump({"model_id": "microsoft/DialoGPT-medium"}, f)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["remove", "microsoft/DialoGPT-medium", "--cache-dir", str(tmp_path), "--force"]
        )

        assert result.exit_code == 0
        assert "removed successfully" in result.output
        assert not model_dir.exists()

    def test_remove_nonexistent_model(self, tmp_path):
        """Test removing a model that doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["remove", "nonexistent/model", "--cache-dir", str(tmp_path)]
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_remove_help(self):
        """Test remove command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["remove", "--help"])

        assert result.exit_code == 0
        assert "Remove a downloaded model" in result.output
        assert "MODEL_ID" in result.output
        assert "--force" in result.output
        assert "--cache-dir" in result.output


class TestFormatSize:
    """Tests for _format_size helper function."""

    def test_format_bytes(self):
        """Test formatting byte sizes."""
        from llm_compress.cli import _format_size

        assert _format_size(0) == "0 B"
        assert _format_size(512) == "512.0 B"
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1536) == "1.5 KB"
        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(1024 * 1024 * 1024) == "1.0 GB"
