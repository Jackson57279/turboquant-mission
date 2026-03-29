"""Benchmark for weight quantization performance and accuracy.

This benchmark measures:
- Compression ratio (theoretical and actual)
- Round-trip accuracy (relative error)
- Quantization/dequantization speed
"""

import time
from pathlib import Path

import torch
import safetensors.torch

from llm_compress.quantization.weight import (
    quantize_tensor,
    dequantize_tensor,
    quantize_model_state_dict,
)


def benchmark_compression_ratio():
    """Measure compression ratio for different tensor sizes."""
    print("=" * 60)
    print("COMPRESSION RATIO BENCHMARK")
    print("=" * 60)
    
    sizes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    
    for size in sizes:
        tensor = torch.randn(*size, dtype=torch.float32)
        original_bytes = tensor.numel() * 4
        
        # 4-bit
        q4, _, _ = quantize_tensor(tensor, bits=4)
        q4_bytes = q4.numel()
        ratio_4bit = original_bytes / q4_bytes
        
        # 8-bit
        q8, _, _ = quantize_tensor(tensor, bits=8)
        q8_bytes = q8.numel()
        ratio_8bit = original_bytes / q8_bytes
        
        print(f"Shape {size}:")
        print(f"  4-bit: {ratio_4bit:.2f}x ({original_bytes/1024/1024:.1f}MB -> {q4_bytes/1024/1024:.1f}MB)")
        print(f"  8-bit: {ratio_8bit:.2f}x ({original_bytes/1024/1024:.1f}MB -> {q8_bytes/1024/1024:.1f}MB)")
    
    print()


def benchmark_accuracy():
    """Measure round-trip accuracy for different bit widths."""
    print("=" * 60)
    print("ACCURACY BENCHMARK")
    print("=" * 60)
    
    # Test with various distributions
    test_tensors = {
        "normal": torch.randn(1024, 1024),
        "small": torch.randn(1024, 1024) * 0.1,
        "large": torch.randn(1024, 1024) * 10.0,
        "mixed": torch.cat([
            torch.randn(512, 1024) * 0.01,
            torch.randn(512, 1024) * 100.0,
        ], dim=0),
    }
    
    print("\nRelative error (lower is better, <1% = >99% accuracy):")
    print("-" * 60)
    
    for name, tensor in test_tensors.items():
        # 4-bit
        q4, s4, shape = quantize_tensor(tensor, bits=4)
        d4 = dequantize_tensor(q4, s4, shape, bits=4)
        error_4bit = (tensor - d4).abs().mean() / tensor.abs().mean()
        
        # 8-bit
        q8, s8, shape = quantize_tensor(tensor, bits=8)
        d8 = dequantize_tensor(q8, s8, shape, bits=8)
        error_8bit = (tensor - d8).abs().mean() / tensor.abs().mean()
        
        print(f"{name:10s}: 4-bit={error_4bit:.4f} ({100*(1-error_4bit):.2f}%), "
              f"8-bit={error_8bit:.4f} ({100*(1-error_8bit):.2f}%)")
    
    print()


def benchmark_speed():
    """Measure quantization/dequantization speed."""
    print("=" * 60)
    print("SPEED BENCHMARK")
    print("=" * 60)
    
    tensor = torch.randn(4096, 4096)
    iterations = 10
    
    # Warmup
    for _ in range(3):
        q, s, shape = quantize_tensor(tensor, bits=4)
        _ = dequantize_tensor(q, s, shape, bits=4)
    
    # 4-bit quantization
    start = time.time()
    for _ in range(iterations):
        q, s, shape = quantize_tensor(tensor, bits=4)
    q4_time = (time.time() - start) / iterations
    
    # 4-bit dequantization
    q, s, shape = quantize_tensor(tensor, bits=4)
    start = time.time()
    for _ in range(iterations):
        _ = dequantize_tensor(q, s, shape, bits=4)
    d4_time = (time.time() - start) / iterations
    
    # 8-bit quantization
    start = time.time()
    for _ in range(iterations):
        q, s, shape = quantize_tensor(tensor, bits=8)
    q8_time = (time.time() - start) / iterations
    
    # 8-bit dequantization
    q, s, shape = quantize_tensor(tensor, bits=8)
    start = time.time()
    for _ in range(iterations):
        _ = dequantize_tensor(q, s, shape, bits=8)
    d8_time = (time.time() - start) / iterations
    
    print(f"Tensor size: 4096x4096 ({4096*4096*4/1024/1024:.1f}MB)")
    print(f"  4-bit quantize:   {q4_time*1000:.1f}ms")
    print(f"  4-bit dequantize: {d4_time*1000:.1f}ms")
    print(f"  8-bit quantize:   {q8_time*1000:.1f}ms")
    print(f"  8-bit dequantize: {d8_time*1000:.1f}ms")
    print()


def benchmark_model_quantization():
    """Benchmark full model state dict quantization."""
    print("=" * 60)
    print("MODEL STATE DICT BENCHMARK")
    print("=" * 60)
    
    # Create a small model-like state dict
    state_dict = {
        f'layer{i}.weight': torch.randn(512, 512)
        for i in range(10)
    }
    state_dict.update({
        f'layer{i}.bias': torch.randn(512)
        for i in range(10)
    })
    
    original_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    
    # Quantize
    start = time.time()
    result_4bit = quantize_model_state_dict(state_dict, bits=4)
    q4_time = time.time() - start
    
    start = time.time()
    result_8bit = quantize_model_state_dict(state_dict, bits=8)
    q8_time = time.time() - start
    
    # Calculate compressed sizes
    q4_size = sum(t.numel() * t.element_size() for t in result_4bit['quantized_tensors'].values())
    q4_size += sum(t.numel() * t.element_size() for t in result_4bit['non_quantized'].values())
    
    q8_size = sum(t.numel() * t.element_size() for t in result_8bit['quantized_tensors'].values())
    q8_size += sum(t.numel() * t.element_size() for t in result_8bit['non_quantized'].values())
    
    print(f"Original size: {original_size/1024/1024:.1f}MB")
    print(f"4-bit: {q4_size/1024/1024:.1f}MB ({original_size/q4_size:.2f}x) in {q4_time*1000:.1f}ms")
    print(f"8-bit: {q8_size/1024/1024:.1f}MB ({original_size/q8_size:.2f}x) in {q8_time*1000:.1f}ms")
    print()


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("WEIGHT QUANTIZATION BENCHMARK")
    print("=" * 60 + "\n")
    
    benchmark_compression_ratio()
    benchmark_accuracy()
    benchmark_speed()
    benchmark_model_quantization()
    
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
