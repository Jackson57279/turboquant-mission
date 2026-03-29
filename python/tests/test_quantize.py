"""Tests for the quantize CLI command."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest
import torch
from click.testing import CliRunner

from llm_compress.cli import main, _format_size
from llm_compress.download import (
    get_cache_dir,
    get_model_dir,
    is_model_cached,
    save_metadata,
)


class TestQuantizeCLI:
    """Tests for the quantize CLI command."""

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    @patch("llm_compress.quantization.weight.get_compression_ratio")
    def test_quantize_success_4bit(self, mock_compression, mock_quantize, mock_cached, tmp_path):
        """Test successful 4-bit quantization."""
        mock_cached.return_value = True
        
        output_dir = tmp_path / "quantized-4bit"
        output_dir.mkdir()
        mock_quantize.return_value = output_dir
        mock_compression.return_value = 3.8
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--bits", "4",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code == 0
        assert "Quantization complete" in result.output
        assert "4-bit" in result.output
        assert "3.80x" in result.output or "3.8x" in result.output
        mock_quantize.assert_called_once_with(
            model_id="microsoft/DialoGPT-medium",
            bits=4,
            cache_dir=str(tmp_path),
        )

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    @patch("llm_compress.quantization.weight.get_compression_ratio")
    def test_quantize_success_8bit(self, mock_compression, mock_quantize, mock_cached, tmp_path):
        """Test successful 8-bit quantization."""
        mock_cached.return_value = True
        
        output_dir = tmp_path / "quantized-8bit"
        output_dir.mkdir()
        mock_quantize.return_value = output_dir
        mock_compression.return_value = 1.9
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--bits", "8",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code == 0
        assert "Quantization complete" in result.output
        assert "8-bit" in result.output
        assert "1.90x" in result.output or "1.9x" in result.output
        mock_quantize.assert_called_once_with(
            model_id="microsoft/DialoGPT-medium",
            bits=8,
            cache_dir=str(tmp_path),
        )

    @patch("llm_compress.download.is_model_cached")
    def test_quantize_without_download_fails(self, mock_cached, tmp_path):
        """Test that quantizing a non-downloaded model fails with clear error."""
        mock_cached.return_value = False
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "nonexistent/model",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "not found" in result.output.lower()
        assert "download it first" in result.output.lower()

    @patch("llm_compress.download.is_model_cached")
    def test_quantize_model_not_downloaded_error(self, mock_cached, tmp_path):
        """Test error message when model is not downloaded (VAL-CLI-015)."""
        mock_cached.return_value = False
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "meta-llama/Llama-2-7b-hf",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code != 0
        # Should show clear error about model not being downloaded
        assert "not found in cache" in result.output.lower() or "download it first" in result.output.lower()

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    @patch("llm_compress.quantization.weight.get_compression_ratio")
    def test_quantize_with_kv_cache_flag(self, mock_compression, mock_quantize, mock_cached, tmp_path):
        """Test quantize with --kv-cache flag (VAL-CLI-006)."""
        mock_cached.return_value = True
        
        output_dir = tmp_path / "quantized-4bit"
        output_dir.mkdir()
        mock_quantize.return_value = output_dir
        mock_compression.return_value = 3.5
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--bits", "4",
            "--kv-cache",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code == 0
        # Should indicate KV cache is enabled
        assert "KV cache quantization enabled" in result.output or "kv cache: enabled" in result.output.lower()

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    @patch("llm_compress.quantization.weight.get_compression_ratio")
    def test_quantize_without_kv_cache(self, mock_compression, mock_quantize, mock_cached, tmp_path):
        """Test quantize without --kv-cache flag shows disabled."""
        mock_cached.return_value = True
        
        output_dir = tmp_path / "quantized-4bit"
        output_dir.mkdir()
        mock_quantize.return_value = output_dir
        mock_compression.return_value = 3.5
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--bits", "4",
            "--no-kv-cache",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code == 0
        # Should indicate KV cache is disabled
        assert "KV cache: disabled" in result.output.lower() or "disabled" in result.output.lower()

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    def test_quantize_shows_progress(self, mock_quantize, mock_cached, tmp_path):
        """Test that quantize shows progress messages (VAL-CLI-004)."""
        mock_cached.return_value = True
        
        output_dir = tmp_path / "quantized-4bit"
        output_dir.mkdir()
        mock_quantize.return_value = output_dir
        
        with patch("llm_compress.quantization.weight.get_compression_ratio") as mock_compression:
            mock_compression.return_value = 3.8
            
            runner = CliRunner()
            result = runner.invoke(main, [
                "quantize",
                "microsoft/DialoGPT-medium",
                "--bits", "4",
                "--cache-dir", str(tmp_path)
            ])
        
        assert result.exit_code == 0
        # Should show progress
        assert "Quantizing" in result.output
        assert "Loading model" in result.output or "preparing" in result.output.lower()

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    @patch("llm_compress.quantization.weight.get_compression_ratio")
    def test_quantize_shows_serve_hint(self, mock_compression, mock_quantize, mock_cached, tmp_path):
        """Test that quantize shows how to serve the quantized model."""
        mock_cached.return_value = True
        
        output_dir = tmp_path / "quantized-4bit"
        output_dir.mkdir()
        mock_quantize.return_value = output_dir
        mock_compression.return_value = 3.8
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code == 0
        assert "To serve this model" in result.output
        assert "llm-compress serve" in result.output

    @patch("llm_compress.download.is_model_cached")
    def test_quantize_invalid_bits(self, mock_cached, tmp_path):
        """Test quantize with invalid bits value fails."""
        mock_cached.return_value = True
        
        runner = CliRunner()
        # Only 4 and 8 are valid choices in Click.Choice
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--bits", "16",  # Invalid choice
            "--cache-dir", str(tmp_path)
        ])
        
        # Click should reject invalid choice
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()

    def test_quantize_help(self):
        """Test quantize command help (VAL-CLI-013)."""
        runner = CliRunner()
        result = runner.invoke(main, ["quantize", "--help"])
        
        assert result.exit_code == 0
        assert "Quantize a downloaded model" in result.output
        assert "--bits" in result.output
        assert "--kv-cache" in result.output
        assert "MODEL_ID" in result.output
        assert "4" in result.output
        assert "8" in result.output

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    def test_quantize_quantization_fails(self, mock_quantize, mock_cached, tmp_path):
        """Test handling when quantization process fails."""
        mock_cached.return_value = True
        mock_quantize.side_effect = RuntimeError("Quantization failed: out of memory")
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code != 0
        assert "Quantization failed" in result.output or "failed" in result.output.lower()

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    @patch("llm_compress.quantization.weight.get_compression_ratio")
    def test_quantize_default_bits_is_4(self, mock_compression, mock_quantize, mock_cached, tmp_path):
        """Test that default bits is 4."""
        mock_cached.return_value = True
        
        output_dir = tmp_path / "quantized-4bit"
        output_dir.mkdir()
        mock_quantize.return_value = output_dir
        mock_compression.return_value = 3.8
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code == 0
        # Verify it was called with bits=4 (the default)
        mock_quantize.assert_called_once()
        call_kwargs = mock_quantize.call_args.kwargs
        assert call_kwargs["bits"] == 4


class TestQuantizeIntegration:
    """Integration tests for quantize command with real model structure."""

    def create_mock_model_files(self, model_dir: Path) -> None:
        """Create mock model files for testing."""
        # Create model.safetensors file (empty but exists)
        (model_dir / "model.safetensors").touch()
        (model_dir / "config.json").write_text('{"model_type": "gpt2"}')
        (model_dir / "tokenizer.json").touch()

    @patch("llm_compress.download.is_model_cached")
    @patch("llm_compress.quantization.weight.quantize_model")
    @patch("llm_compress.quantization.weight.get_compression_ratio")
    def test_quantize_e2e_mocked_backend(self, mock_compression, mock_quantize, mock_cached, tmp_path):
        """End-to-end test for quantization with mocked backend."""
        mock_cached.return_value = True
        
        # Setup model directory
        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        original_dir = model_dir / "original"
        original_dir.mkdir(parents=True)
        
        # Create metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({"model_id": "microsoft/DialoGPT-medium"}, f)
        
        self.create_mock_model_files(original_dir)
        
        # Mock quantization output
        quantized_dir = model_dir / "quantized-4bit"
        quantized_dir.mkdir()
        mock_quantize.return_value = quantized_dir
        mock_compression.return_value = 3.8
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "quantize",
            "microsoft/DialoGPT-medium",
            "--bits", "4",
            "--cache-dir", str(tmp_path)
        ])
        
        assert result.exit_code == 0
        assert "Quantization complete" in result.output
        assert "4-bit" in result.output
        mock_quantize.assert_called_once_with(
            model_id="microsoft/DialoGPT-medium",
            bits=4,
            cache_dir=str(tmp_path),
        )
