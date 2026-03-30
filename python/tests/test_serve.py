"""Tests for the serve CLI command."""
from __future__ import annotations

from contextlib import suppress
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from llm_compress.cli import main
from llm_compress.download import save_metadata
from llm_compress.server.app import _server_state


class MockBackend:
    """Mock backend for testing serve command."""

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.config = kwargs
        self._initialized = False
        self.backend_name = kwargs.get("backend_name", "mock")

    def initialize(self):
        self._initialized = True

    def health(self):
        return {
            "status": "healthy",
            "backend": self.backend_name,
            "model_id": self.model_id,
            "initialized": self._initialized,
        }

    def generate(self, prompt, **kwargs):
        return {
            "id": "test-gen-123",
            "choices": [{"text": "Mock response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    def chat(self, messages, **kwargs):
        return {
            "id": "test-chat-123",
            "choices": [{"message": {"role": "assistant", "content": "Mock response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    def shutdown(self):
        self._initialized = False


class TestServeCommand:
    """Tests for the serve CLI command."""

    def setup_method(self):
        """Reset server state before each test."""
        _server_state["backend"] = None
        _server_state["model_id"] = None
        _server_state["backend_name"] = None
        _server_state["cache_dir"] = None

    def teardown_method(self):
        """Clean up server state after each test."""
        if _server_state["backend"] is not None:
            with suppress(Exception):
                _server_state["backend"].shutdown()
        _server_state["backend"] = None

    @patch("uvicorn.run")
    @patch("llm_compress.server.app.create_app")
    def test_serve_default_port(self, mock_create_app, mock_uvicorn_run, tmp_path):
        """VAL-CLI-007: Serve command starts API server on port 3200."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        runner = CliRunner()
        result = runner.invoke(main, [
            "serve",
            "microsoft/DialoGPT-medium",
            "--cache-dir", str(tmp_path)
        ])

        assert result.exit_code == 0
        call_kwargs = mock_create_app.call_args.kwargs
        assert call_kwargs["model_id"] == "microsoft/DialoGPT-medium"
        assert call_kwargs["backend"] == "vllm"
        assert call_kwargs["enable_kv_compression"] is True

        # Verify uvicorn.run was called with default port 3200
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args.kwargs
        assert call_kwargs["port"] == 3200
        assert call_kwargs["host"] == "127.0.0.1"

    @patch("uvicorn.run")
    @patch("llm_compress.server.app.create_app")
    def test_serve_custom_port(self, mock_create_app, mock_uvicorn_run, tmp_path):
        """VAL-CLI-010: Serve command with custom port."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        runner = CliRunner()
        result = runner.invoke(main, [
            "serve",
            "microsoft/DialoGPT-medium",
            "--port", "8080",
            "--cache-dir", str(tmp_path)
        ])

        assert result.exit_code == 0
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args.kwargs
        assert call_kwargs["port"] == 8080

    @patch("uvicorn.run")
    @patch("llm_compress.server.app.create_app")
    def test_serve_backend_vllm(self, mock_create_app, mock_uvicorn_run, tmp_path):
        """VAL-CLI-008: Serve command with vLLM backend."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        runner = CliRunner()
        result = runner.invoke(main, [
            "serve",
            "microsoft/DialoGPT-medium",
            "--backend", "vllm",
            "--cache-dir", str(tmp_path)
        ])

        assert result.exit_code == 0
        assert "Backend: vllm" in result.output
        call_kwargs = mock_create_app.call_args.kwargs
        assert call_kwargs["backend"] == "vllm"

    @patch("uvicorn.run")
    @patch("llm_compress.server.app.create_app")
    def test_serve_backend_llama_cpp(self, mock_create_app, mock_uvicorn_run, tmp_path):
        """VAL-CLI-009: Serve command with llama.cpp backend."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        runner = CliRunner()
        result = runner.invoke(main, [
            "serve",
            "microsoft/DialoGPT-medium",
            "--backend", "llama-cpp",
            "--cache-dir", str(tmp_path)
        ])

        assert result.exit_code == 0
        assert "Backend: llama-cpp" in result.output
        call_kwargs = mock_create_app.call_args.kwargs
        assert call_kwargs["backend"] == "llama-cpp"

    @patch("llm_compress.download.is_model_cached")
    @patch("uvicorn.run")
    @patch("llm_compress.server.app.create_app")
    def test_serve_unquantized_model_warning(self, mock_create_app, mock_uvicorn_run, mock_cached, tmp_path):
        """VAL-CLI-016: Serving a non-quantized model shows a performance warning."""
        mock_cached.return_value = True
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        model_dir.mkdir()
        save_metadata("microsoft/DialoGPT-medium", {"model_id": "microsoft/DialoGPT-medium"}, cache_dir=str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(main, [
            "serve",
            "microsoft/DialoGPT-medium",
            "--cache-dir", str(tmp_path)
        ])

        assert result.exit_code == 0
        assert "Warning" in result.output or "unquantized" in result.output.lower()
        assert "Performance may be reduced" in result.output or "consider quantizing" in result.output.lower()

    @patch("llm_compress.download.is_model_cached")
    @patch("uvicorn.run")
    @patch("llm_compress.server.app.create_app")
    def test_serve_quantized_model_no_warning(self, mock_create_app, mock_uvicorn_run, mock_cached, tmp_path):
        """Test that serving a quantized model does not show unquantized warning."""
        mock_cached.return_value = True
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        model_dir = tmp_path / "microsoft--DialoGPT-medium"
        model_dir.mkdir()
        quantized_dir = model_dir / "quantized-4bit"
        quantized_dir.mkdir()
        save_metadata(
            "microsoft/DialoGPT-medium",
            {
                "model_id": "microsoft/DialoGPT-medium",
                "quantized": True,
                "quantization": {"bits": 4, "type": "nf4"}
            },
            cache_dir=str(tmp_path)
        )

        runner = CliRunner()
        result = runner.invoke(main, [
            "serve",
            "microsoft/DialoGPT-medium",
            "--cache-dir", str(tmp_path)
        ])

        assert result.exit_code == 0
        assert "unquantized" not in result.output.lower() or "Serving quantized model" in result.output

    @patch("uvicorn.run")
    @patch("llm_compress.server.app.create_app")
    def test_serve_shows_endpoints(self, mock_create_app, mock_uvicorn_run, tmp_path):
        """Test that serve command shows available endpoints."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        runner = CliRunner()
        result = runner.invoke(main, [
            "serve",
            "microsoft/DialoGPT-medium",
            "--cache-dir", str(tmp_path)
        ])

        assert result.exit_code == 0
        assert "/health" in result.output
        assert "/v1/models" in result.output
        assert "/v1/chat/completions" in result.output
        assert "/v1/completions" in result.output

    def test_serve_help(self):
        """VAL-CLI-013: CLI help command shows serve options."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])

        assert result.exit_code == 0
        assert "Start the OpenAI-compatible API server" in result.output
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--backend" in result.output
        assert "--kv-cache" in result.output
        assert "MODEL_ID" in result.output
        assert "vllm" in result.output
        assert "llama-cpp" in result.output
