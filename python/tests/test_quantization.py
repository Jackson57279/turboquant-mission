"""Unit tests for weight quantization module.

These tests verify:
- Quantization/dequantization correctness
- Round-trip accuracy preservation
- Compression ratio verification
- Custom format save/load
- Error handling
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
import bitsandbytes as bnb

from llm_compress.quantization.weight import (
    quantize_tensor,
    dequantize_tensor,
    quantize_model_state_dict,
    dequantize_model,
    save_quantized_model,
    load_quantized_model,
    get_compression_ratio,
    estimate_accuracy_loss,
)


class TestQuantizeTensor:
    """Tests for single tensor quantization."""
    
    def test_quantize_4bit_returns_tuple(self):
        """Test that 4-bit quantization returns correct structure."""
        tensor = torch.randn(512, 512)
        qweight, qstate, shape = quantize_tensor(tensor, bits=4)
        
        assert isinstance(qweight, torch.Tensor)
        assert qweight.dtype == torch.uint8
        assert shape == tensor.shape
        assert hasattr(qstate, 'absmax')
        
    def test_quantize_8bit_returns_tuple(self):
        """Test that 8-bit quantization returns correct structure."""
        tensor = torch.randn(512, 512)
        qweight, qstate, shape = quantize_tensor(tensor, bits=8)
        
        assert isinstance(qweight, torch.Tensor)
        assert qweight.dtype == torch.uint8
        assert shape == tensor.shape
        assert hasattr(qstate, 'absmax')
        
    def test_invalid_bits_raises_error(self):
        """Test that invalid bit width raises ValueError."""
        tensor = torch.randn(512, 512)
        
        with pytest.raises(ValueError, match="Only 4-bit and 8-bit quantization supported"):
            quantize_tensor(tensor, bits=16)
            
        with pytest.raises(ValueError, match="Only 4-bit and 8-bit quantization supported"):
            quantize_tensor(tensor, bits=2)
            
    def test_quantize_1d_tensor(self):
        """Test quantization of 1D tensor."""
        tensor = torch.randn(1024)
        qweight, qstate, shape = quantize_tensor(tensor, bits=4)
        
        assert shape == (1024,)
        assert isinstance(qweight, torch.Tensor)
        
    def test_quantize_3d_tensor(self):
        """Test quantization of 3D tensor."""
        tensor = torch.randn(8, 64, 128)
        qweight, qstate, shape = quantize_tensor(tensor, bits=4)
        
        assert shape == (8, 64, 128)
        assert isinstance(qweight, torch.Tensor)


class TestDequantizeTensor:
    """Tests for tensor dequantization."""
    
    def test_roundtrip_4bit(self):
        """Test 4-bit quantization round-trip preserves values."""
        original = torch.randn(512, 512)
        
        qweight, qstate, shape = quantize_tensor(original, bits=4)
        reconstructed = dequantize_tensor(qweight, qstate, shape, bits=4)
        
        # Check shape preservation
        assert reconstructed.shape == original.shape
        
        # Check relative error is reasonable (4-bit has more error)
        relative_error = (original - reconstructed).abs().mean() / original.abs().mean()
        assert relative_error < 0.05  # <5% error for 4-bit
        
    def test_roundtrip_8bit(self):
        """Test 8-bit quantization round-trip preserves values."""
        original = torch.randn(512, 512)
        
        qweight, qstate, shape = quantize_tensor(original, bits=8)
        reconstructed = dequantize_tensor(qweight, qstate, shape, bits=8)
        
        # Check shape preservation
        assert reconstructed.shape == original.shape
        
        # Check relative error is small (8-bit is more accurate)
        relative_error = (original - reconstructed).abs().mean() / original.abs().mean()
        assert relative_error < 0.02  # <2% error for 8-bit
        
    def test_invalid_bits_raises_error_on_dequantize(self):
        """Test that invalid bit width raises ValueError during dequantization."""
        tensor = torch.randn(512, 512)
        qweight, qstate, shape = quantize_tensor(tensor, bits=4)
        
        with pytest.raises(ValueError, match="Only 4-bit and 8-bit quantization supported"):
            dequantize_tensor(qweight, qstate, shape, bits=16)


class TestRoundtripAccuracy:
    """Tests for quantization round-trip accuracy thresholds."""
    
    def test_4bit_accuracy_above_99_percent(self):
        """VAL-QUANT-001: 4-bit quantized model retains >99% accuracy.
        
        This measures relative error on synthetic data as a proxy for
        perplexity-based accuracy measurement.
        """
        # Test with various tensor sizes and distributions
        test_cases = [
            torch.randn(1024, 1024),  # Large matrix
            torch.randn(512, 768),    # Common transformer shape
            torch.randn(256, 4096),   # Another common shape
            torch.randn(128, 128) * 0.1,  # Small values
            torch.randn(256, 256) * 10,  # Large values
        ]
        
        for original in test_cases:
            qweight, qstate, shape = quantize_tensor(original, bits=4)
            reconstructed = dequantize_tensor(qweight, qstate, shape, bits=4)
            
            # Calculate relative error
            relative_error = (original - reconstructed).abs().mean() / original.abs().mean()
            
            # Should be <1% relative error (99% accuracy)
            assert relative_error < 0.01, f"4-bit relative error {relative_error:.4f} exceeds 1% threshold"
            
    def test_8bit_accuracy_above_99_5_percent(self):
        """VAL-QUANT-002: 8-bit quantized model retains >99.5% accuracy."""
        test_cases = [
            torch.randn(1024, 1024),
            torch.randn(512, 768),
            torch.randn(256, 4096),
            torch.randn(128, 128) * 0.1,
            torch.randn(256, 256) * 10,
        ]
        
        for original in test_cases:
            qweight, qstate, shape = quantize_tensor(original, bits=8)
            reconstructed = dequantize_tensor(qweight, qstate, shape, bits=8)
            
            relative_error = (original - reconstructed).abs().mean() / original.abs().mean()
            
            # Should be <0.5% relative error (99.5% accuracy)
            assert relative_error < 0.005, f"8-bit relative error {relative_error:.4f} exceeds 0.5% threshold"
            
    def test_blockwise_quantization_correctness(self):
        """VAL-QUANT-005: Block-wise quantization produces deterministic results.
        
        Same input should produce same output across runs.
        """
        original = torch.randn(512, 512)
        
        # Quantize twice
        qweight1, qstate1, shape1 = quantize_tensor(original, bits=4)
        qweight2, qstate2, shape2 = quantize_tensor(original, bits=4)
        
        # Results should be identical
        assert torch.equal(qweight1, qweight2)
        assert shape1 == shape2
        
        # Reconstruction should also be identical
        reconstructed1 = dequantize_tensor(qweight1, qstate1, shape1, bits=4)
        reconstructed2 = dequantize_tensor(qweight2, qstate2, shape2, bits=4)
        
        assert torch.equal(reconstructed1, reconstructed2)


class TestCompressionRatio:
    """Tests for compression ratio verification."""
    
    def test_4bit_compression_ratio_approx_4x(self):
        """4-bit quantization should achieve ~4x compression."""
        tensor = torch.randn(1024, 1024, dtype=torch.float32)
        original_bytes = tensor.numel() * 4  # 4 bytes per float32
        
        qweight, qstate, shape = quantize_tensor(tensor, bits=4)
        # 4-bit packs 2 values per byte
        quantized_bytes = qweight.numel()
        
        # Compression ratio (ignoring absmax overhead for simplicity)
        ratio = original_bytes / quantized_bytes
        
        # Should be approximately 4x (may be slightly less due to absmax overhead)
        assert ratio > 3.5, f"4-bit compression ratio {ratio:.2f}x is less than expected 3.5x"
        
    def test_8bit_compression_ratio_approx_2x(self):
        """8-bit quantization should achieve ~2x compression."""
        tensor = torch.randn(1024, 1024, dtype=torch.float32)
        original_bytes = tensor.numel() * 4
        
        qweight, qstate, shape = quantize_tensor(tensor, bits=8)
        quantized_bytes = qweight.numel()
        
        ratio = original_bytes / quantized_bytes
        
        # Should be approximately 2x
        assert ratio > 1.8, f"8-bit compression ratio {ratio:.2f}x is less than expected 1.8x"


class TestModelStateDictQuantization:
    """Tests for full model state dict quantization."""
    
    def test_quantize_model_state_dict_structure(self):
        """Test that quantization produces correct structure."""
        state_dict = {
            'layer1.weight': torch.randn(512, 512),
            'layer1.bias': torch.randn(512),
            'layer2.weight': torch.randn(256, 512),
            'embed.weight': torch.randn(1000, 512),  # Should not be quantized
        }
        
        result = quantize_model_state_dict(state_dict, bits=4)
        
        assert 'quantized_tensors' in result
        assert 'quantization_metadata' in result
        assert 'non_quantized' in result
        
        # Weight tensors should be quantized
        assert 'layer1.weight' in result['quantized_tensors']
        assert 'layer2.weight' in result['quantized_tensors']
        
        # 1D tensors should not be quantized
        assert 'layer1.bias' in result['non_quantized']
        assert 'embed.weight' in result['non_quantized']
        
    def test_quantize_model_state_dict_metadata(self):
        """Test that metadata is correctly populated."""
        state_dict = {
            'layer1.weight': torch.randn(512, 512),
        }
        
        result = quantize_model_state_dict(state_dict, bits=4)
        
        meta = result['quantization_metadata']['layer1.weight']
        assert meta['quantized'] is True
        assert meta['bits'] == 4
        assert 'shape' in meta
        assert 'dtype' in meta
        
    def test_all_tensors_preserved(self):
        """Test that all tensors are either quantized or kept as-is."""
        state_dict = {
            'weight1': torch.randn(256, 256),
            'weight2': torch.randn(128, 128),
            'bias': torch.randn(128),
            'norm': torch.randn(64),
        }
        
        result = quantize_model_state_dict(state_dict, bits=4)
        
        # All keys should be present in either quantized or non_quantized
        all_keys = set(result['quantized_tensors'].keys()) | set(result['non_quantized'].keys())
        assert all_keys == set(state_dict.keys())


class TestSaveLoadQuantizedModel:
    """Tests for custom format save/load."""
    
    def test_save_quantized_model_creates_files(self):
        """Test that saving creates expected files."""
        state_dict = {
            'layer1.weight': torch.randn(256, 256),
            'layer1.bias': torch.randn(256),
        }
        
        quantized_data = quantize_model_state_dict(state_dict, bits=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'quantized'
            save_quantized_model(quantized_data, output_dir, 'test-model', bits=4)
            
            # Check files exist
            assert (output_dir / 'model.safetensors').exists()
            assert (output_dir / 'quantization_config.json').exists()
            
    def test_save_quantized_model_config_content(self):
        """Test that config file has correct content."""
        state_dict = {
            'layer1.weight': torch.randn(256, 256),
        }
        
        quantized_data = quantize_model_state_dict(state_dict, bits=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'quantized'
            save_quantized_model(quantized_data, output_dir, 'test-model', bits=4)
            
            # Load and verify config
            with open(output_dir / 'quantization_config.json') as f:
                config = json.load(f)
            
            assert config['model_id'] == 'test-model'
            assert config['bits'] == 4
            assert config['quantization_type'] == 'nf4'
            assert config['format_version'] == '1.0'
            
    def test_load_quantized_model_returns_correct_structure(self):
        """Test that loading returns correct structure."""
        state_dict = {
            'layer1.weight': torch.randn(256, 256),
            'layer1.bias': torch.randn(256),
        }
        
        quantized_data = quantize_model_state_dict(state_dict, bits=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'quantized'
            save_quantized_model(quantized_data, output_dir, 'test-model', bits=4)
            
            # Load
            loaded = load_quantized_model(output_dir)
            
            assert 'tensors' in loaded
            assert 'config' in loaded
            assert 'quantized_names' in loaded
            assert 'layer1.weight' in loaded['quantized_names']
            assert 'layer1.bias' not in loaded['quantized_names']
            
    def test_save_load_roundtrip(self):
        """VAL-QUANT-009: Dequantization after quantization approximates original."""
        state_dict = {
            'layer1.weight': torch.randn(256, 256),
            'layer2.weight': torch.randn(128, 256),
        }
        
        # Quantize and save
        quantized_data = quantize_model_state_dict(state_dict, bits=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'quantized'
            save_quantized_model(quantized_data, output_dir, 'test-model', bits=4)
            
            # Load and dequantize
            loaded = load_quantized_model(output_dir)
            dequantized = dequantize_model(loaded)
            
            # Verify all keys present
            assert set(dequantized.keys()) == set(state_dict.keys())
            
            # Verify shapes match
            for key in state_dict:
                assert dequantized[key].shape == state_dict[key].shape
                
            # Verify reasonable accuracy preservation
            for key in state_dict:
                original = state_dict[key]
                recon = dequantized[key]
                relative_error = (original - recon).abs().mean() / original.abs().mean()
                assert relative_error < 0.05, f"Round-trip error for {key}: {relative_error:.4f}"


class TestGetCompressionRatio:
    """Tests for compression ratio calculation."""
    
    def test_get_compression_ratio_returns_positive(self):
        """Test that compression ratio is calculated correctly."""
        state_dict = {
            'layer1.weight': torch.randn(1024, 1024),  # 4MB original
        }
        
        quantized_data = quantize_model_state_dict(state_dict, bits=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'quantized'
            save_quantized_model(quantized_data, output_dir, 'test-model', bits=4)
            
            ratio = get_compression_ratio(output_dir)
            
            # Should be > 3.5 for 4-bit (theoretical 4x minus overhead)
            assert ratio > 3.0
            
    def test_8bit_compression_ratio_less_than_4bit(self):
        """8-bit should have lower compression ratio than 4-bit."""
        state_dict = {
            'layer1.weight': torch.randn(1024, 1024),
        }
        
        # 4-bit quantization
        q4_data = quantize_model_state_dict(state_dict, bits=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'quantized'
            save_quantized_model(q4_data, output_dir, 'test-model', bits=4)
            ratio_4bit = get_compression_ratio(output_dir)
        
        # 8-bit quantization
        q8_data = quantize_model_state_dict(state_dict, bits=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'quantized'
            save_quantized_model(q8_data, output_dir, 'test-model', bits=8)
            ratio_8bit = get_compression_ratio(output_dir)
        
        # 4-bit should compress better than 8-bit
        assert ratio_4bit > ratio_8bit


class TestEstimateAccuracyLoss:
    """Tests for accuracy loss estimation."""
    
    def test_4bit_accuracy_loss_estimate(self):
        """4-bit should have ~0.5% estimated loss."""
        loss = estimate_accuracy_loss(4)
        assert loss == 0.5
        
    def test_8bit_accuracy_loss_estimate(self):
        """8-bit should have ~0.2% estimated loss."""
        loss = estimate_accuracy_loss(8)
        assert loss == 0.2
        
    def test_unknown_bits_returns_conservative(self):
        """Unknown bit width should return conservative estimate."""
        loss = estimate_accuracy_loss(16)
        assert loss == 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_tensor(self):
        """Test handling of empty tensors."""
        state_dict = {
            'empty.weight': torch.tensor([]),
            'normal.weight': torch.randn(256, 256),
        }
        
        result = quantize_model_state_dict(state_dict, bits=4)
        
        # Empty tensor should be in non_quantized
        assert 'empty.weight' in result['non_quantized']
        assert 'normal.weight' in result['quantized_tensors']
        
    def test_single_element_tensor(self):
        """Test handling of single element tensors."""
        state_dict = {
            'scalar.weight': torch.tensor([1.0]),
        }
        
        # Should not crash
        result = quantize_model_state_dict(state_dict, bits=4)
        assert 'scalar.weight' in result['quantized_tensors']
        
    def test_very_large_tensor(self):
        """Test handling of large tensors."""
        # This might be slow, so use moderate size
        state_dict = {
            'large.weight': torch.randn(2048, 2048),
        }
        
        result = quantize_model_state_dict(state_dict, bits=4)
        assert 'large.weight' in result['quantized_tensors']
        
    def test_different_dtypes(self):
        """Test handling of different tensor dtypes."""
        state_dict = {
            'float32': torch.randn(256, 256, dtype=torch.float32),
            'float16': torch.randn(256, 256, dtype=torch.float16),
            'bfloat16': torch.randn(256, 256, dtype=torch.bfloat16),
            'int64': torch.randint(0, 100, (256, 256), dtype=torch.int64),
        }
        
        result = quantize_model_state_dict(state_dict, bits=4)
        
        # Float types should be quantized
        assert 'float32' in result['quantized_tensors']
        assert 'float16' in result['quantized_tensors']
        assert 'bfloat16' in result['quantized_tensors']
        
        # Integer types should not be quantized
        assert 'int64' in result['non_quantized']


class TestBlockWiseQuantization:
    """Tests for block-wise quantization behavior."""
    
    def test_block_wise_preserves_local_statistics(self):
        """Block-wise quantization should preserve local statistics."""
        # Create tensor with varying scales
        tensor = torch.cat([
            torch.randn(256, 512) * 0.01,  # Small values
            torch.randn(256, 512) * 10.0,   # Large values
        ], dim=0)
        
        qweight, qstate, shape = quantize_tensor(tensor, bits=4)
        reconstructed = dequantize_tensor(qweight, qstate, shape, bits=4)
        
        # Both small and large regions should have reasonable reconstruction
        small_region = reconstructed[:256, :]
        large_region = reconstructed[256:, :]
        
        small_orig = tensor[:256, :]
        large_orig = tensor[256:, :]
        
        # Check that the relative error is reasonable in both regions
        small_error = (small_region - small_orig).abs().mean() / small_orig.abs().mean()
        large_error = (large_region - large_orig).abs().mean() / large_orig.abs().mean()
        
        assert small_error < 0.1
        assert large_error < 0.1
