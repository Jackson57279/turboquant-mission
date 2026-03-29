"""Unit tests for vLLM backend adapter with TurboQuant KV cache integration.

These tests verify:
- Backend initialization with and without vLLM
- TurboQuant KV cache manager functionality
- Health check returns correct status
- Error handling for missing vLLM
- Compression ratio tracking
- Monkey-patch application for attention layers
"""

import pytest
import torch
from unittest.mock import MagicMock, Mock, patch

from llm_compress.backends.vllm import (
    VLLM_AVAILABLE,
    VLLMBackend,
    VLLMBackendStub,
    TurboQuantKVCacheManager,
    TurboQuantAttentionWrapper,
)


class TestTurboQuantKVCacheManager:
    """Tests for TurboQuant KV cache manager."""
    
    def test_manager_initialization(self):
        """Test KV cache manager initialization."""
        manager = TurboQuantKVCacheManager(
            head_dim=64,
            num_layers=12,
            key_bits=3,
            value_bits=2,
        )
        
        assert manager.head_dim == 64
        assert manager.num_layers == 12
        assert manager.key_bits == 3
        assert manager.value_bits == 2
        assert manager.key_proj_dim == 32  # default: head_dim // 2
        assert manager.value_group_size == 8
        assert not manager._fitted
        assert manager.compression_ratio == 0.0
    
    def test_manager_fit(self):
        """Test fitting manager on sample data."""
        manager = TurboQuantKVCacheManager(head_dim=64, num_layers=12)
        
        sample_keys = torch.randn(100, 64)
        sample_values = torch.randn(100, 64)
        
        manager.fit(sample_keys, sample_values)
        
        assert manager._fitted
        assert manager.quantizer._fitted
    
    def test_compress_and_decompress_layer(self):
        """Test compressing and decompressing layer cache."""
        manager = TurboQuantKVCacheManager(head_dim=64, num_layers=12)
        
        # Create sample KV cache
        batch_size = 2
        num_heads = 4
        seq_len = 32
        head_dim = 64
        
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Compress
        compressed = manager.compress_layer_cache(0, keys, values)
        
        assert 'key_indices' in compressed
        assert 'key_codebook' in compressed
        assert 'value_indices' in compressed
        assert 'value_codebook' in compressed
        assert 'seq_len' in compressed
        assert 'metadata' in compressed
        
        # Verify compression ratio is calculated
        assert manager.compression_ratio > 0
        
        # Decompress
        recovered_keys, recovered_values = manager.decompress_layer_cache(0)
        
        assert recovered_keys.shape == keys.shape
        assert recovered_values.shape == values.shape
    
    def test_free_layer_cache(self):
        """Test freeing layer cache memory."""
        manager = TurboQuantKVCacheManager(head_dim=64, num_layers=12)
        
        keys = torch.randn(2, 4, 32, 64)
        values = torch.randn(2, 4, 32, 64)
        
        manager.compress_layer_cache(0, keys, values)
        assert 0 in manager.compressed_cache
        
        manager.free_layer_cache(0)
        assert 0 not in manager.compressed_cache
    
    def test_free_all_cache(self):
        """Test freeing all cache memory."""
        manager = TurboQuantKVCacheManager(head_dim=64, num_layers=12)
        
        # Compress multiple layers
        for layer_idx in range(3):
            keys = torch.randn(2, 4, 32, 64)
            values = torch.randn(2, 4, 32, 64)
            manager.compress_layer_cache(layer_idx, keys, values)
        
        assert len(manager.compressed_cache) == 3
        
        manager.free_all_cache()
        
        assert len(manager.compressed_cache) == 0
        assert len(manager.layer_shapes) == 0
    
    def test_decompress_missing_layer_raises_error(self):
        """Test that decompressing missing layer raises error."""
        manager = TurboQuantKVCacheManager(head_dim=64, num_layers=12)
        
        with pytest.raises(KeyError, match="No compressed cache found for layer 5"):
            manager.decompress_layer_cache(5)
    
    def test_attention_with_compressed_keys(self):
        """Test attention computation with compressed keys."""
        manager = TurboQuantKVCacheManager(head_dim=64, num_layers=12)
        
        # Create and compress cache
        keys = torch.randn(2, 4, 32, 64)
        values = torch.randn(2, 4, 32, 64)
        manager.compress_layer_cache(0, keys, values)
        
        # Create query
        query = torch.randn(2, 4, 16, 64)
        
        # Compute attention
        output = manager.compute_attention_with_compressed_keys(query, 0)
        
        assert output.shape == (2, 4, 16, 64)


class TestTurboQuantAttentionWrapper:
    """Tests for TurboQuant attention wrapper."""
    
    def test_wrapper_initialization(self):
        """Test attention wrapper initialization."""
        original_attn = MagicMock()
        kv_manager = TurboQuantKVCacheManager(head_dim=64, num_layers=12)
        
        wrapper = TurboQuantAttentionWrapper(original_attn, kv_manager, layer_idx=3)
        
        assert wrapper.original_attention == original_attn
        assert wrapper.kv_manager == kv_manager
        assert wrapper.layer_idx == 3
    
    def test_wrapper_forward(self):
        """Test attention wrapper forward pass."""
        original_attn = MagicMock()
        kv_manager = TurboQuantKVCacheManager(head_dim=64, num_layers=12)
        wrapper = TurboQuantAttentionWrapper(original_attn, kv_manager, layer_idx=0)
        
        # Create input tensors
        query = torch.randn(2, 4, 16, 64)
        key = torch.randn(2, 4, 32, 64)
        value = torch.randn(2, 4, 32, 64)
        
        # Run forward
        output = wrapper.forward(query, key, value)
        
        # Verify output shape
        assert output.shape == (2, 4, 16, 64)
        
        # Verify cache was compressed
        assert 0 in wrapper.kv_manager.compressed_cache


class TestVLLMBackendStub:
    """Tests for vLLM backend stub (when vLLM not available)."""
    
    def test_stub_initialization(self):
        """Test stub initialization."""
        stub = VLLMBackendStub("test-model")
        
        assert stub.model_id == "test-model"
    
    def test_stub_initialize_raises_error(self):
        """Test that stub initialize raises informative error."""
        stub = VLLMBackendStub("test-model")
        
        with pytest.raises(RuntimeError, match="vLLM is not installed"):
            stub.initialize()
    
    def test_stub_health_returns_unhealthy(self):
        """Test stub health returns unhealthy status."""
        stub = VLLMBackendStub("test-model")
        
        health = stub.health()
        
        assert health["status"] == "unhealthy"
        assert health["backend"] == "vllm"
        assert health["vllm_available"] is False
        assert health["initialized"] is False
        assert "error" in health
    
    def test_stub_generate_raises_error(self):
        """Test that stub generate raises error."""
        stub = VLLMBackendStub("test-model")
        
        with pytest.raises(RuntimeError, match="vLLM is not installed"):
            stub.generate("test prompt")
    
    def test_stub_chat_raises_error(self):
        """Test that stub chat raises error."""
        stub = VLLMBackendStub("test-model")
        
        with pytest.raises(RuntimeError, match="vLLM is not installed"):
            stub.chat([{"role": "user", "content": "hello"}])
    
    def test_stub_shutdown_is_noop(self):
        """Test that stub shutdown is a no-op."""
        stub = VLLMBackendStub("test-model")
        
        # Should not raise
        stub.shutdown()
    
    def test_stub_free_kv_cache_is_noop(self):
        """Test that stub free_kv_cache is a no-op."""
        stub = VLLMBackendStub("test-model")
        
        # Should not raise
        stub.free_kv_cache()


class TestVLLMBackend:
    """Tests for vLLM backend (mocked)."""
    
    @pytest.fixture
    def mock_vllm_env(self):
        """Fixture to mock vLLM environment."""
        with patch('llm_compress.backends.vllm.VLLM_AVAILABLE', True):
            with patch('llm_compress.backends.vllm.LLM') as mock_llm_class:
                with patch('llm_compress.backends.vllm.SamplingParams') as mock_sampling:
                    yield mock_llm_class, mock_sampling
    
    def test_backend_initialization(self, mock_vllm_env):
        """Test backend initialization with mocked vLLM."""
        mock_llm_class, _ = mock_vllm_env
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            backend = VLLMBackend(
                "microsoft/DialoGPT-medium",
                enable_kv_compression=True,
            )
            
            backend.initialize()
            
            assert backend._initialized is True
            assert backend.llm is not None
            assert backend.kv_manager is not None
    
    def test_backend_without_kv_compression(self, mock_vllm_env):
        """Test backend initialization without KV compression."""
        mock_llm_class, _ = mock_vllm_env
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            backend = VLLMBackend(
                "microsoft/DialoGPT-medium",
                enable_kv_compression=False,
            )
            
            backend.initialize()
            
            assert backend._initialized is True
            assert backend.kv_manager is None
    
    def test_backend_health_healthy(self, mock_vllm_env):
        """Test health check when backend is healthy."""
        mock_llm_class, _ = mock_vllm_env
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            backend = VLLMBackend("microsoft/DialoGPT-medium")
            backend.initialize()
            
            health = backend.health()
            
            assert health["status"] == "healthy"
            assert health["backend"] == "vllm"
            assert health["vllm_available"] is True
            assert health["initialized"] is True
            assert health["kv_compression_enabled"] is True
    
    def test_backend_health_uninitialized(self, mock_vllm_env):
        """Test health check when backend not initialized."""
        with patch('llm_compress.backends.vllm.VLLM_VERSION', '0.18.0'):
            backend = VLLMBackend("microsoft/DialoGPT-medium")
            # Don't initialize
            
            health = backend.health()
            
            assert health["status"] == "unhealthy"
            assert health["initialized"] is False
            assert "error" in health
    
    def test_backend_generate_sync(self, mock_vllm_env):
        """Test synchronous generation."""
        mock_llm_class, mock_sampling = mock_vllm_env
        
        # Create mock output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated text", finish_reason="stop")]
        mock_output.prompt_token_ids = [1, 2, 3]
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            backend = VLLMBackend("microsoft/DialoGPT-medium")
            backend.initialize()
            
            result = backend.generate("Test prompt", max_tokens=50)
            
            assert "choices" in result
            assert len(result["choices"]) == 1
            assert result["choices"][0]["text"] == "Generated text"
            assert result["choices"][0]["finish_reason"] == "stop"
            assert "usage" in result
    
    def test_backend_generate_stream(self, mock_vllm_env):
        """Test streaming generation."""
        mock_llm_class, mock_sampling = mock_vllm_env
        
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Hello world", finish_reason="stop")]
        mock_output.prompt_token_ids = [1, 2, 3]
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            backend = VLLMBackend("microsoft/DialoGPT-medium")
            backend.initialize()
            
            chunks = list(backend.generate("Test prompt", stream=True))
            
            assert len(chunks) > 0
            # First chunks should have content
            assert all("choices" in chunk for chunk in chunks)
    
    def test_backend_chat_sync(self, mock_vllm_env):
        """Test synchronous chat completion."""
        mock_llm_class, _ = mock_vllm_env
        
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Hello! How can I help?", finish_reason="stop")]
        mock_output.prompt_token_ids = [1, 2, 3]
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            backend = VLLMBackend("microsoft/DialoGPT-medium")
            backend.initialize()
            
            messages = [
                {"role": "user", "content": "Hello"},
            ]
            result = backend.chat(messages)
            
            assert "choices" in result
            assert result["choices"][0]["message"]["role"] == "assistant"
            assert "Hello! How can I help?" in result["choices"][0]["message"]["content"]
    
    def test_backend_format_chat_messages(self):
        """Test chat message formatting."""
        with patch('llm_compress.backends.vllm.VLLM_AVAILABLE', True):
            backend = VLLMBackend("test-model")
            
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
            
            prompt = backend._format_chat_messages(messages)
            
            assert "System: You are helpful" in prompt
            assert "User: Hello" in prompt
            assert "Assistant: Hi there" in prompt
            assert "Assistant:" in prompt  # Final prompt for response
    
    def test_backend_generate_without_initialize_raises(self, mock_vllm_env):
        """Test that generate raises error if not initialized."""
        with patch('llm_compress.backends.vllm.VLLM_VERSION', '0.18.0'):
            backend = VLLMBackend("test-model")
            # Don't initialize
            
            with pytest.raises(RuntimeError, match="Backend not initialized"):
                backend.generate("Test prompt")
    
    def test_backend_chat_without_initialize_raises(self, mock_vllm_env):
        """Test that chat raises error if not initialized."""
        with patch('llm_compress.backends.vllm.VLLM_VERSION', '0.18.0'):
            backend = VLLMBackend("test-model")
            # Don't initialize
            
            with pytest.raises(RuntimeError, match="Backend not initialized"):
                backend.chat([{"role": "user", "content": "hello"}])
    
    def test_backend_shutdown(self, mock_vllm_env):
        """Test backend shutdown."""
        mock_llm_class, _ = mock_vllm_env
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            backend = VLLMBackend("microsoft/DialoGPT-medium")
            backend.initialize()
            
            assert backend._initialized is True
            assert backend.kv_manager is not None
            
            backend.shutdown()
            
            assert backend._initialized is False
            assert backend.kv_manager is None
    
    def test_backend_free_kv_cache(self, mock_vllm_env):
        """Test free_kv_cache method."""
        mock_llm_class, _ = mock_vllm_env
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            with patch('llm_compress.backends.vllm.free_kv_cache') as mock_free:
                backend = VLLMBackend("microsoft/DialoGPT-medium")
                backend.initialize()
                
                # Add some compressed cache
                backend.kv_manager.compress_layer_cache(
                    0,
                    torch.randn(2, 4, 32, 64),
                    torch.randn(2, 4, 32, 64),
                )
                
                backend.free_kv_cache()
                
                assert len(backend.kv_manager.compressed_cache) == 0
                mock_free.assert_called_once()
    
    def test_backend_config_passthrough(self, mock_vllm_env):
        """Test that config options are passed through correctly."""
        mock_llm_class, _ = mock_vllm_env
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
            )
            
            backend = VLLMBackend(
                "microsoft/DialoGPT-medium",
                temperature=0.5,
                top_p=0.95,
                max_model_len=2048,
                gpu_memory_utilization=0.8,
                dtype="float16",
                seed=123,
            )
            
            assert backend.temperature == 0.5
            assert backend.top_p == 0.95
            assert backend.max_model_len == 2048
            assert backend.gpu_memory_utilization == 0.8
            assert backend.dtype == "float16"
            assert backend.seed == 123


class TestVLLMIntegration:
    """Integration tests for vLLM backend."""
    
    def test_backend_registration_with_registry(self):
        """Test that backend is properly registered."""
        from llm_compress.backends.registry import get_backend, list_backends
        
        # vLLM should be in the list of backends
        backends = list_backends()
        assert "vllm" in backends
        
        # Should be able to get backend instance (will be stub if vLLM not available)
        backend = get_backend("vllm", "test-model")
        assert backend is not None
        
        if not VLLM_AVAILABLE:
            assert isinstance(backend, VLLMBackendStub)
    
    def test_kv_cache_compression_active(self):
        """Test that KV cache compression is active during operations."""
        manager = TurboQuantKVCacheManager(
            head_dim=64,
            num_layers=12,
            key_bits=3,
            value_bits=2,
        )
        
        # Compress multiple layers
        for layer_idx in range(5):
            keys = torch.randn(2, 4, 64, 64)
            values = torch.randn(2, 4, 64, 64)
            manager.compress_layer_cache(layer_idx, keys, values)
        
        # Verify compression is tracked
        assert manager.compression_ratio > 0
        
        # Verify all layers are cached
        assert len(manager.compressed_cache) == 5
        
        # Compute attention with compressed keys
        query = torch.randn(2, 4, 16, 64)
        output = manager.compute_attention_with_compressed_keys(query, 0)
        
        assert output.shape == (2, 4, 16, 64)
    
    def test_error_handling_missing_vllm(self):
        """Test error handling when vLLM is not installed."""
        with patch('llm_compress.backends.vllm.VLLM_AVAILABLE', False):
            from llm_compress.backends.vllm import VLLMBackendStub
            
            stub = VLLMBackendStub("test-model")
            
            with pytest.raises(RuntimeError, match="vLLM is not installed"):
                stub.initialize()


# Run baseline tests if available
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
class TestVLLMWithRealInstallation:
    """Tests that require actual vLLM installation."""
    
    def test_vllm_imports(self):
        """Test that vLLM can be imported."""
        import vllm
        
        assert hasattr(vllm, "LLM")
        assert hasattr(vllm, "SamplingParams")
    
    def test_backend_health_without_model(self):
        """Test health check without loading a model."""
        backend = VLLMBackend("test-model")
        
        health = backend.health()
        
        assert health["vllm_available"] is True
        assert health["status"] == "unhealthy"  # Not initialized yet
