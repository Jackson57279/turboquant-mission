"""Unit tests for llama.cpp backend adapter with GGUF conversion support.

These tests verify:
- Backend initialization with and without llama-cpp-python
- GGUF conversion from HuggingFace format
- Support for quantized GGUF models
- Backend health check implementation
- Generate and chat functionality
- Streaming support
- Error handling for missing dependencies
"""

import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest

from llm_compress.backends.llama_cpp import (
    LLAMA_CPP_AVAILABLE,
    GGUFConverter,
    LlamaCppBackend,
    LlamaCppBackendStub,
)


class TestGGUFConverter:
    """Tests for GGUF conversion functionality."""

    def test_converter_initialization(self):
        """Test GGUF converter initialization."""
        converter = GGUFConverter(
            model_id="microsoft/DialoGPT-medium",
            quantization_type="Q4_K_M",
        )
        
        assert converter.model_id == "microsoft/DialoGPT-medium"
        assert converter.quantization_type == "Q4_K_M"
        assert os.path.exists(converter.output_dir)

    def test_quantization_mapping(self):
        """Test quantization type mapping."""
        converter = GGUFConverter("test/model")
        
        assert converter._map_quantization_to_outtype() == "q4_k_m"
        assert converter._map_quantization_to_outtype("Q8_0") == "q8_0"
        assert converter._map_quantization_to_outtype("F16") == "f16"
        assert converter._map_quantization_to_outtype("Q4_0") == "q4_0"

    @patch("llm_compress.backends.llama_cpp.list_repo_files")
    @patch("llm_compress.backends.llama_cpp.hf_hub_download")
    def test_find_preconverted_gguf_success(self, mock_download, mock_list_files):
        """Test finding pre-converted GGUF file."""
        mock_list_files.return_value = [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
        ]
        mock_download.return_value = "/tmp/model-Q4_K_M.gguf"
        
        converter = GGUFConverter("microsoft/DialoGPT-medium", quantization_type="Q4_K_M")
        result = converter.find_preconverted_gguf()
        
        assert result == "/tmp/model-Q4_K_M.gguf"
        mock_download.assert_called_once()

    @patch("llm_compress.backends.llama_cpp.list_repo_files")
    def test_find_preconverted_gguf_not_found(self, mock_list_files):
        """Test when no pre-converted GGUF is found."""
        mock_list_files.side_effect = Exception("Repo not found")
        
        converter = GGUFConverter("unknown/model")
        result = converter.find_preconverted_gguf()
        
        assert result is None

    def test_find_preconverted_without_hf_available(self):
        """Test that None is returned when HF not available."""
        with patch("llm_compress.backends.llama_cpp.HF_AVAILABLE", False):
            converter = GGUFConverter("test/model")
            assert converter.find_preconverted_gguf() is None


class TestLlamaCppBackendStub:
    """Tests for llama.cpp backend stub (when llama-cpp-python not available)."""

    def test_stub_initialization(self):
        """Test stub initialization."""
        stub = LlamaCppBackendStub("test-model")
        
        assert stub.model_id == "test-model"

    def test_stub_initialize_raises_error(self):
        """Test that stub initialize raises informative error."""
        stub = LlamaCppBackendStub("test-model")
        
        with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
            stub.initialize()

    def test_stub_health_returns_unhealthy(self):
        """Test stub health returns unhealthy status."""
        stub = LlamaCppBackendStub("test-model")
        
        health = stub.health()
        
        assert health["status"] == "unhealthy"
        assert health["backend"] == "llama.cpp"
        assert health["llama_cpp_available"] is False
        assert "error" in health

    def test_stub_generate_raises_error(self):
        """Test that stub generate raises error."""
        stub = LlamaCppBackendStub("test-model")
        
        with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
            stub.generate("test prompt")

    def test_stub_chat_raises_error(self):
        """Test that stub chat raises error."""
        stub = LlamaCppBackendStub("test-model")
        
        with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
            stub.chat([{"role": "user", "content": "hello"}])

    def test_stub_shutdown_is_noop(self):
        """Test that stub shutdown is a no-op."""
        stub = LlamaCppBackendStub("test-model")
        
        # Should not raise
        stub.shutdown()


class TestLlamaCppBackend:
    """Tests for llama.cpp backend (mocked)."""

    @pytest.fixture
    def mock_llama_cpp_env(self):
        """Fixture to mock llama.cpp environment."""
        with patch("llm_compress.backends.llama_cpp.LLAMA_CPP_AVAILABLE", True):
            with patch("llm_compress.backends.llama_cpp.Llama") as mock_llama_class:
                yield mock_llama_class

    def test_backend_initialization_with_gguf_path(self, mock_llama_cpp_env):
        """Test backend initialization with direct GGUF path."""
        mock_llama_class = mock_llama_cpp_env
        mock_llm_instance = MagicMock()
        mock_llama_class.return_value = mock_llm_instance
        
        with patch("os.path.exists", return_value=True):
            backend = LlamaCppBackend(
                model_id="/path/to/model.gguf",
                n_ctx=4096,
                n_threads=8,
            )
            backend.initialize()
            
            assert backend._initialized is True
            assert backend.llm is not None
            mock_llama_class.assert_called_once()
            
            # Verify initialization parameters
            call_kwargs = mock_llama_class.call_args[1]
            assert call_kwargs["model_path"] == "/path/to/model.gguf"
            assert call_kwargs["n_ctx"] == 4096
            assert call_kwargs["n_threads"] == 8

    def test_backend_initialization_with_hf_model(self, mock_llama_cpp_env):
        """Test backend initialization with HuggingFace model ID."""
        mock_llama_class = mock_llama_cpp_env
        mock_llm_instance = MagicMock()
        mock_llama_class.return_value = mock_llm_instance
        
        with patch("llm_compress.backends.llama_cpp.HF_AVAILABLE", True):
            with patch.object(GGUFConverter, "find_preconverted_gguf", return_value="/tmp/model.gguf"):
                backend = LlamaCppBackend(
                    model_id="microsoft/DialoGPT-medium",
                    quantization="Q4_K_M",
                )
                backend.initialize()
                
                assert backend._initialized is True
                assert backend.llm is not None

    def test_backend_initialization_fails_without_hf(self, mock_llama_cpp_env):
        """Test that initialization fails when HF not available for conversion."""
        with patch("llm_compress.backends.llama_cpp.HF_AVAILABLE", False):
            backend = LlamaCppBackend(
                model_id="microsoft/DialoGPT-medium",  # Not a GGUF path
                quantization="Q4_K_M",
            )
            
            with pytest.raises(RuntimeError, match="HuggingFace libraries required"):
                backend.initialize()

    def test_backend_initialization_fails_without_llama_cpp(self):
        """Test that initialization fails when llama-cpp-python not available."""
        with patch("llm_compress.backends.llama_cpp.LLAMA_CPP_AVAILABLE", False):
            backend = LlamaCppBackend("test-model")
            
            with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
                backend.initialize()

    def test_backend_health_healthy(self, mock_llama_cpp_env):
        """Test health check when backend is healthy."""
        mock_llama_class = mock_llama_cpp_env
        mock_llm_instance = MagicMock()
        mock_llm_instance.n_vocab.return_value = 32000
        mock_llm_instance.n_ctx.return_value = 2048
        mock_llama_class.return_value = mock_llm_instance
        
        with patch("os.path.exists", return_value=True):
            backend = LlamaCppBackend("/path/to/model.gguf")
            backend.initialize()
            
            health = backend.health()
            
            assert health["status"] == "healthy"
            assert health["backend"] == "llama.cpp"
            assert health["llama_cpp_available"] is True
            assert health["initialized"] is True
            assert health["vocab_size"] == 32000
            assert health["context_size"] == 2048

    def test_backend_health_uninitialized(self, mock_llama_cpp_env):
        """Test health check when backend not initialized."""
        backend = LlamaCppBackend("test-model")
        # Don't initialize
        
        health = backend.health()
        
        assert health["status"] == "unhealthy"
        assert health["initialized"] is False
        assert "error" in health

    def test_backend_generate_sync(self, mock_llama_cpp_env):
        """Test synchronous generation."""
        mock_llama_class = mock_llama_cpp_env
        mock_llm_instance = MagicMock()
        mock_llm_instance.create_completion.return_value = {
            "choices": [{"text": "Generated text", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_llama_class.return_value = mock_llm_instance
        
        with patch("os.path.exists", return_value=True):
            backend = LlamaCppBackend("/path/to/model.gguf")
            backend.initialize()
            
            result = backend.generate("Test prompt", max_tokens=50)
            
            assert "choices" in result
            assert len(result["choices"]) == 1
            assert result["choices"][0]["text"] == "Generated text"
            assert result["choices"][0]["finish_reason"] == "stop"
            assert "usage" in result

    def test_backend_generate_stream(self, mock_llama_cpp_env):
        """Test streaming generation."""
        mock_llama_class = mock_llama_cpp_env
        mock_llm_instance = MagicMock()
        
        # Mock streaming output
        def mock_stream():
            yield {"choices": [{"text": "Hello"}]}
            yield {"choices": [{"text": " world"}]}
        
        mock_llm_instance.create_completion.return_value = mock_stream()
        mock_llama_class.return_value = mock_llm_instance
        
        with patch("os.path.exists", return_value=True):
            backend = LlamaCppBackend("/path/to/model.gguf")
            backend.initialize()
            
            chunks = list(backend.generate("Test prompt", stream=True))
            
            assert len(chunks) == 3  # 2 content chunks + 1 final chunk
            assert all("choices" in chunk for chunk in chunks)

    def test_backend_chat_sync(self, mock_llama_cpp_env):
        """Test synchronous chat completion."""
        mock_llama_class = mock_llama_cpp_env
        mock_llm_instance = MagicMock()
        mock_llm_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help?"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_llama_class.return_value = mock_llm_instance
        
        with patch("os.path.exists", return_value=True):
            backend = LlamaCppBackend("/path/to/model.gguf")
            backend.initialize()
            
            messages = [{"role": "user", "content": "Hello"}]
            result = backend.chat(messages)
            
            assert "choices" in result
            assert result["choices"][0]["message"]["role"] == "assistant"
            assert "Hello! How can I help?" in result["choices"][0]["message"]["content"]

    def test_backend_chat_stream(self, mock_llama_cpp_env):
        """Test streaming chat completion."""
        mock_llama_class = mock_llama_cpp_env
        mock_llm_instance = MagicMock()
        
        # Mock streaming output
        def mock_stream():
            yield {"choices": [{"delta": {"role": "assistant", "content": "Hi"}}]}
            yield {"choices": [{"delta": {"content": " there"}}]}
        
        mock_llm_instance.create_chat_completion.return_value = mock_stream()
        mock_llama_class.return_value = mock_llm_instance
        
        with patch("os.path.exists", return_value=True):
            backend = LlamaCppBackend("/path/to/model.gguf")
            backend.initialize()
            
            chunks = list(backend.chat([{"role": "user", "content": "Hello"}], stream=True))
            
            assert len(chunks) == 3  # 2 content chunks + 1 final chunk
            assert all("choices" in chunk for chunk in chunks)

    def test_backend_generate_without_initialize_raises(self, mock_llama_cpp_env):
        """Test that generate raises error if not initialized."""
        backend = LlamaCppBackend("test-model")
        # Don't initialize
        
        with pytest.raises(RuntimeError, match="Backend not initialized"):
            backend.generate("Test prompt")

    def test_backend_chat_without_initialize_raises(self, mock_llama_cpp_env):
        """Test that chat raises error if not initialized."""
        backend = LlamaCppBackend("test-model")
        # Don't initialize
        
        with pytest.raises(RuntimeError, match="Backend not initialized"):
            backend.chat([{"role": "user", "content": "hello"}])

    def test_backend_shutdown(self, mock_llama_cpp_env):
        """Test backend shutdown."""
        mock_llama_class = mock_llama_cpp_env
        mock_llm_instance = MagicMock()
        mock_llama_class.return_value = mock_llm_instance
        
        with patch("os.path.exists", return_value=True):
            backend = LlamaCppBackend("/path/to/model.gguf")
            backend.initialize()
            
            assert backend._initialized is True
            assert backend.llm is not None
            
            backend.shutdown()
            
            assert backend._initialized is False
            assert backend.llm is None

    def test_backend_config_passthrough(self, mock_llama_cpp_env):
        """Test that config options are passed through correctly."""
        backend = LlamaCppBackend(
            "test-model",
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=20,
            seed=42,
            verbose=True,
        )
        
        assert backend.n_ctx == 4096
        assert backend.n_threads == 8
        assert backend.n_gpu_layers == 20
        assert backend.seed == 42
        assert backend.verbose is True


class TestLlamaCppIntegration:
    """Integration tests for llama.cpp backend."""

    def test_backend_registration_with_registry(self):
        """Test that backend is properly registered."""
        from llm_compress.backends.registry import get_backend, list_backends
        
        # llama.cpp should be in the list of backends
        backends = list_backends()
        assert "llama-cpp" in backends
        
        # Should be able to get backend instance (will be stub if llama-cpp not available)
        backend = get_backend("llama-cpp", "test-model")
        assert backend is not None
        
        if not LLAMA_CPP_AVAILABLE:
            assert isinstance(backend, LlamaCppBackendStub)

    def test_error_handling_missing_llama_cpp(self):
        """Test error handling when llama-cpp-python is not installed."""
        with patch("llm_compress.backends.llama_cpp.LLAMA_CPP_AVAILABLE", False):
            from llm_compress.backends.llama_cpp import LlamaCppBackendStub
            
            stub = LlamaCppBackendStub("test-model")
            
            with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
                stub.initialize()

    def test_direct_gguf_path_detection(self):
        """Test detection of direct GGUF file paths."""
        # Test .gguf extension detection
        backend = LlamaCppBackend("/path/to/model.gguf")
        assert backend._is_direct_gguf is True
        
        # Test with gguf_path parameter
        backend2 = LlamaCppBackend("test-model", gguf_path="/path/to/model.gguf")
        assert backend2._is_direct_gguf is True
        
        # Test HuggingFace model ID (not a GGUF path)
        backend3 = LlamaCppBackend("microsoft/DialoGPT-medium")
        assert backend3._is_direct_gguf is False


# Run baseline tests if available
@pytest.mark.skipif(not LLAMA_CPP_AVAILABLE, reason="llama-cpp-python not installed")
class TestLlamaCppWithRealInstallation:
    """Tests that require actual llama-cpp-python installation."""

    def test_llama_cpp_imports(self):
        """Test that llama-cpp-python can be imported."""
        from llama_cpp import Llama
        
        assert hasattr(Llama, "create_completion")
        assert hasattr(Llama, "create_chat_completion")

    def test_backend_health_without_model(self):
        """Test health check without loading a model."""
        backend = LlamaCppBackend("test-model")
        
        health = backend.health()
        
        assert health["llama_cpp_available"] is True
        assert health["status"] == "unhealthy"  # Not initialized yet
