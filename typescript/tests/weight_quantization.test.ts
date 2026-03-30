/**
 * Weight quantization tests.
 *
 * Tests for 4-bit and 8-bit block-wise weight quantization.
 */

import { describe, it, expect } from 'bun:test';
import {
  quantizeTensor,
  dequantizeTensor,
  calculateCompressionRatio,
  estimateAccuracyLoss,
} from '../src/quantization/weight.js';

describe('Weight Quantization', () => {
  describe('4-bit quantization', () => {
    it('should quantize and dequantize tensor correctly', () => {
      // Create test tensor with known values
      const size = 256;
      const tensor = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        tensor[i] = Math.sin(i * 0.1) * 0.5; // Values in [-0.5, 0.5]
      }

      const shape = [16, 16];
      const quantized = quantizeTensor(tensor, 4, shape);

      // Verify quantization structure
      expect(quantized.data).toBeInstanceOf(Uint8Array);
      expect(quantized.metadata.bits).toBe(4);
      expect(quantized.metadata.shape).toEqual(shape);
      expect(quantized.metadata.quantized).toBe(true);
      expect(quantized.metadata.absmax).toBeDefined();
      expect(quantized.metadata.absmax.length).toBeGreaterThan(0);

      // Verify dequantization
      const dequantized = dequantizeTensor(quantized.data, quantized.metadata);
      expect(dequantized).toBeInstanceOf(Float32Array);
      expect(dequantized.length).toBe(tensor.length);

      // Check that dequantized values are close to original
      let maxError = 0;
      let sumSquaredError = 0;
      for (let i = 0; i < size; i++) {
        const error = Math.abs(dequantized[i] - tensor[i]);
        maxError = Math.max(maxError, error);
        sumSquaredError += error * error;
      }

      const mse = sumSquaredError / size;
      expect(maxError).toBeLessThan(0.15); // Max error should be reasonable
      expect(mse).toBeLessThan(0.01); // MSE should be small
    });

    it('should achieve ~8x compression ratio for 4-bit', () => {
      const tensor = new Float32Array(1024).map(() => Math.random() - 0.5);
      const shape = [32, 32];
      const quantized = quantizeTensor(tensor, 4, shape);

      // Original: 1024 * 4 bytes = 4096 bytes
      // Quantized: 1024/2 bytes (packed) + metadata
      // Should achieve approximately 8x compression
      const originalSize = tensor.length * 4; // float32 = 4 bytes
      const quantizedSize = quantized.data.length;
      const ratio = calculateCompressionRatio(originalSize, quantizedSize);

      expect(ratio).toBeGreaterThan(6); // At least 6x compression
      expect(ratio).toBeLessThan(10); // But not more than 10x (due to metadata)
    });

    it('should handle empty tensors gracefully', () => {
      const tensor = new Float32Array(0);
      const shape = [0, 0];
      const quantized = quantizeTensor(tensor, 4, shape);

      expect(quantized.data.length).toBe(0);
      expect(quantized.metadata.shape).toEqual(shape);
    });

    it('should preserve accuracy >99% for well-behaved data', () => {
      // Create data with normal distribution (well-suited for NF4)
      const size = 512;
      const tensor = new Float32Array(size);

      // Box-Muller transform for normal distribution
      for (let i = 0; i < size; i += 2) {
        const u1 = Math.random();
        const u2 = Math.random();
        const radius = Math.sqrt(-2 * Math.log(u1));
        const theta = 2 * Math.PI * u2;
        tensor[i] = radius * Math.cos(theta) * 0.3;
        if (i + 1 < size) {
          tensor[i + 1] = radius * Math.sin(theta) * 0.3;
        }
      }

      const shape = [32, 16];
      const quantized = quantizeTensor(tensor, 4, shape);
      const dequantized = dequantizeTensor(quantized.data, quantized.metadata);

      // Calculate signal-to-noise ratio
      let signalPower = 0;
      let noisePower = 0;
      for (let i = 0; i < size; i++) {
        signalPower += tensor[i] * tensor[i];
        const noise = dequantized[i] - tensor[i];
        noisePower += noise * noise;
      }

      const snr = 10 * Math.log10(signalPower / noisePower);
      expect(snr).toBeGreaterThan(20); // SNR > 20dB indicates good quality
    });

    it('should throw error for invalid bit width', () => {
      const tensor = new Float32Array(64);
      expect(() => quantizeTensor(tensor, 2 as 4, [8, 8])).toThrow();
      expect(() => quantizeTensor(tensor, 16 as 4, [8, 8])).toThrow();
    });

    it('should handle large tensors efficiently', () => {
      const size = 4096;
      const tensor = new Float32Array(size).map(() => Math.random() - 0.5);
      const shape = [64, 64];

      const start = performance.now();
      const quantized = quantizeTensor(tensor, 4, shape);
      const quantTime = performance.now() - start;

      const start2 = performance.now();
      dequantizeTensor(quantized.data, quantized.metadata);
      const dequantTime = performance.now() - start2;

      // Should complete in reasonable time (< 1 second for 4096 elements)
      expect(quantTime).toBeLessThan(1000);
      expect(dequantTime).toBeLessThan(1000);
    });
  });

  describe('8-bit quantization', () => {
    it('should quantize and dequantize tensor correctly', () => {
      const size = 256;
      const tensor = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        tensor[i] = Math.cos(i * 0.05) * 0.8;
      }

      const shape = [16, 16];
      const quantized = quantizeTensor(tensor, 8, shape);

      expect(quantized.data).toBeInstanceOf(Uint8Array);
      expect(quantized.metadata.bits).toBe(8);
      expect(quantized.metadata.shape).toEqual(shape);
      expect(quantized.data.length).toBe(size); // 1 byte per element for 8-bit

      const dequantized = dequantizeTensor(quantized.data, quantized.metadata);
      expect(dequantized.length).toBe(tensor.length);

      // 8-bit should have better accuracy than 4-bit
      let maxError = 0;
      let sumSquaredError = 0;
      for (let i = 0; i < size; i++) {
        const error = Math.abs(dequantized[i] - tensor[i]);
        maxError = Math.max(maxError, error);
        sumSquaredError += error * error;
      }

      const mse = sumSquaredError / size;
      expect(maxError).toBeLessThan(0.05); // Tighter bound for 8-bit
      expect(mse).toBeLessThan(0.001);
    });

    it('should achieve ~4x compression ratio for 8-bit', () => {
      const tensor = new Float32Array(1024).map(() => Math.random() - 0.5);
      const shape = [32, 32];
      const quantized = quantizeTensor(tensor, 8, shape);

      const originalSize = tensor.length * 4;
      const quantizedSize = quantized.data.length;
      const ratio = calculateCompressionRatio(originalSize, quantizedSize);

      expect(ratio).toBeGreaterThan(3); // At least 3x compression
      expect(ratio).toBeLessThan(5); // But not more than 5x
    });

    it('should have higher accuracy than 4-bit', () => {
      const size = 512;
      const tensor = new Float32Array(size).map(() => Math.random() * 2 - 1);
      const shape = [32, 16];

      const quantized4 = quantizeTensor(tensor, 4, shape);
      const quantized8 = quantizeTensor(tensor, 8, shape);

      const dequantized4 = dequantizeTensor(quantized4.data, quantized4.metadata);
      const dequantized8 = dequantizeTensor(quantized8.data, quantized8.metadata);

      let error4 = 0;
      let error8 = 0;
      for (let i = 0; i < size; i++) {
        error4 += Math.abs(dequantized4[i] - tensor[i]);
        error8 += Math.abs(dequantized8[i] - tensor[i]);
      }

      // 8-bit should have lower total error
      expect(error8).toBeLessThan(error4);
    });
  });

  describe('estimateAccuracyLoss', () => {
    it('should return conservative estimates', () => {
      const loss4bit = estimateAccuracyLoss(4);
      const loss8bit = estimateAccuracyLoss(8);

      expect(loss4bit).toBe(0.5); // 0.5% for 4-bit
      expect(loss8bit).toBe(0.2); // 0.2% for 8-bit
      expect(loss4bit).toBeGreaterThan(loss8bit);
    });
  });

  describe('block-wise quantization', () => {
    it('should use correct block size from metadata', () => {
      const tensor = new Float32Array(128).fill(1.0);
      const shape = [8, 16];
      const quantized = quantizeTensor(tensor, 4, shape);

      expect(quantized.metadata.blocksize).toBeDefined();
      expect(quantized.metadata.blocksize).toBeGreaterThan(0);

      // absmax should have correct number of blocks
      const expectedBlocks = Math.ceil(tensor.length / (quantized.metadata.blocksize || 64));
      expect(quantized.metadata.absmax.length).toBe(expectedBlocks);
    });

    it('should handle tensors smaller than block size', () => {
      const tensor = new Float32Array(32).map(() => Math.random());
      const shape = [4, 8];
      const quantized = quantizeTensor(tensor, 4, shape);

      expect(quantized.metadata.absmax.length).toBe(1);
      expect(quantized.data.length).toBeGreaterThan(0);
    });
  });
});
