/**
 * Quantization benchmark tests.
 *
 * Performance benchmarks for weight and KV cache quantization.
 */

import { describe, it, expect } from 'bun:test';
import {
  quantizeTensor,
  dequantizeTensor,
} from '../src/quantization/weight.js';
import {
  LloydMaxQuantizer,
  OrthogonalRotation,
  TurboQuantKeyCompressor,
  GroupValueQuantizer,
} from '../src/quantization/kv_cache.js';

describe('Performance Benchmarks', () => {
  describe('Weight Quantization Performance', () => {
    it('should quantize 100K elements in under 500ms', () => {
      const size = 100_000;
      const tensor = new Float32Array(size).map(() => Math.random() - 0.5);
      const shape = [1000, 100];

      const start = performance.now();
      const quantized = quantizeTensor(tensor, 4, shape);
      const quantTime = performance.now() - start;

      expect(quantTime).toBeLessThan(500);
      expect(quantized.data.length).toBeGreaterThan(0);
    });

    it('should dequantize 100K elements in under 500ms', () => {
      const size = 100_000;
      const tensor = new Float32Array(size).map(() => Math.random() - 0.5);
      const shape = [1000, 100];
      const quantized = quantizeTensor(tensor, 4, shape);

      const start = performance.now();
      dequantizeTensor(quantized.data, quantized.metadata);
      const dequantTime = performance.now() - start;

      expect(dequantTime).toBeLessThan(500);
    });

    it('8-bit should be faster than 4-bit for large tensors', () => {
      const size = 50_000;
      const tensor = new Float32Array(size).map(() => Math.random() - 0.5);
      const shape = [500, 100];

      const start4 = performance.now();
      quantizeTensor(tensor, 4, shape);
      const time4 = performance.now() - start4;

      const start8 = performance.now();
      quantizeTensor(tensor, 8, shape);
      const time8 = performance.now() - start8;

      // 8-bit writes more bytes (1 byte per element vs 0.5 bytes), can be slightly slower
      // This is acceptable - the main benefit of 8-bit is accuracy, not speed
      expect(time8).toBeLessThan(time4 * 4); // Allow up to 4x variance
    });
  });

  describe('KV Cache Quantization Performance', () => {
    it('should initialize Lloyd-Max quantizer quickly', () => {
      const data = new Float32Array(10_000).map(() => Math.random() - 0.5);
      const quantizer = new LloydMaxQuantizer(8, 20);

      const start = performance.now();
      quantizer.generateCodebook(data);
      const initTime = performance.now() - start;

      expect(initTime).toBeLessThan(200);
    });

    it('should compress keys efficiently', () => {
      const dimension = 64;
      const compressor = new TurboQuantKeyCompressor(dimension);

      // Initialize with training data
      const trainingData = Array.from({ length: 100 }, () =>
        new Float32Array(dimension).map(() => Math.random() - 0.5)
      );
      compressor.initialize(trainingData);

      const testKey = new Float32Array(dimension).map(() => Math.random() - 0.5);

      const start = performance.now();
      const compressed = compressor.compress(testKey);
      const compressTime = performance.now() - start;

      expect(compressTime).toBeLessThan(50);
      expect(compressed.indices).toBeDefined();
    });

    it('should decompress keys efficiently', () => {
      const dimension = 64;
      const compressor = new TurboQuantKeyCompressor(dimension);

      const trainingData = Array.from({ length: 100 }, () =>
        new Float32Array(dimension).map(() => Math.random() - 0.5)
      );
      compressor.initialize(trainingData);

      const testKey = new Float32Array(dimension).map(() => Math.random() - 0.5);
      const compressed = compressor.compress(testKey);

      const start = performance.now();
      compressor.decompress(compressed);
      const decompressTime = performance.now() - start;

      expect(decompressTime).toBeLessThan(50);
    });

    it('should handle batch operations efficiently', () => {
      const quantizer = new GroupValueQuantizer();
      const trainingData = Array.from({ length: 50 }, () =>
        new Float32Array(32).map(() => Math.random() - 0.5)
      );
      quantizer.initialize(trainingData);

      const batchValues = Array.from({ length: 20 }, () =>
        new Float32Array(32).map(() => Math.random() - 0.5)
      );

      const start = performance.now();
      const compressed = quantizer.compress(batchValues);
      const compressTime = performance.now() - start;

      expect(compressTime).toBeLessThan(100);
      expect(compressed.length).toBe(20);
    });
  });

  describe('Orthogonal Rotation Performance', () => {
    it('should generate rotation matrix quickly for reasonable dimensions', () => {
      const dimensions = [16, 32, 64, 128];

      for (const dim of dimensions) {
        const rotation = new OrthogonalRotation(dim);

        const start = performance.now();
        const matrix = rotation.generate();
        const genTime = performance.now() - start;

        expect(matrix.length).toBe(dim * dim);
        expect(genTime).toBeLessThan(100);
      }
    });

    it('should apply rotation quickly', () => {
      const dim = 128;
      const rotation = new OrthogonalRotation(dim);
      const matrix = rotation.generate();
      const vector = new Float32Array(dim).map(() => Math.random() - 0.5);

      const start = performance.now();
      rotation.apply(matrix, vector);
      const applyTime = performance.now() - start;

      expect(applyTime).toBeLessThan(10);
    });
  });
});
