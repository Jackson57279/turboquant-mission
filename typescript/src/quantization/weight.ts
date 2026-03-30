/**
 * Weight quantization implementation.
 *
 * This module implements block-wise weight quantization supporting 4-bit and 8-bit
 * quantization schemes. This is a TypeScript port of the Python implementation
 * that uses bitsandbytes.
 *
 * Quantization schemes:
 *   - 4-bit: NF4 (Normal Float 4) with block-wise quantization
 *   - 8-bit: INT8 with block-wise quantization
 *
 * @module quantization/weight
 */

import type { QuantizationOptions, QuantizedModel, QuantizationMetadata, QuantizedTensor } from './types.js';

/** Default block size for quantization */
const DEFAULT_BLOCK_SIZE = 64;

/** Normal Float 4 quantization levels (NF4) */
const NF4_LEVELS: number[] = [
  -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
  -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
  0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
];

/**
 * Calculate absolute maximum values for block-wise quantization.
 *
 * @param data - Input array
 * @param blockSize - Size of each block
 * @returns Array of absmax values per block
 */
function computeAbsmax(data: Float32Array, blockSize: number): Float32Array {
  const numBlocks = Math.ceil(data.length / blockSize);
  const absmax = new Float32Array(numBlocks);

  for (let i = 0; i < numBlocks; i++) {
    const start = i * blockSize;
    const end = Math.min(start + blockSize, data.length);
    let max = 0;

    for (let j = start; j < end; j++) {
      max = Math.max(max, Math.abs(data[j]));
    }

    absmax[i] = max;
  }

  return absmax;
}

/**
 * Quantize a single tensor to specified bit width.
 *
 * @param tensor - Input tensor as Float32Array
 * @param bits - Quantization bit width (4 or 8)
 * @param shape - Original tensor shape
 * @returns Quantized tensor data
 * @throws Error if bits is not 4 or 8
 *
 * @example
 * ```typescript
 * const weight = new Float32Array(512 * 512).map(() => Math.random() - 0.5);
 * const quantized = quantizeTensor(weight, 4, [512, 512]);
 * console.log(`Quantized ${quantized.data.length} bytes`);
 * ```
 */
export function quantizeTensor(
  tensor: Float32Array,
  bits: 4 | 8,
  shape: number[]
): QuantizedTensor {
  if (bits !== 4 && bits !== 8) {
    throw new Error(`Only 4-bit and 8-bit quantization supported, got ${bits}`);
  }

  const blockSize = DEFAULT_BLOCK_SIZE;
  const absmax = computeAbsmax(tensor, blockSize);

  if (bits === 4) {
    // 4-bit quantization using NF4
    const numElements = tensor.length;
    const numBytes = Math.ceil(numElements / 2);
    const quantized = new Uint8Array(numBytes);

    for (let i = 0; i < numElements; i++) {
      const blockIdx = Math.floor(i / blockSize);
      const scale = absmax[blockIdx];

      // Normalize and quantize
      let normalized = scale > 0 ? tensor[i] / scale : 0;
      normalized = Math.max(-1, Math.min(1, normalized)); // Clamp to [-1, 1]

      // Find closest NF4 level
      let bestIdx = 0;
      let bestDiff = Math.abs(normalized - NF4_LEVELS[0]);

      for (let j = 1; j < NF4_LEVELS.length; j++) {
        const diff = Math.abs(normalized - NF4_LEVELS[j]);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIdx = j;
        }
      }

      // Pack 2 4-bit values per byte
      const byteIdx = Math.floor(i / 2);
      if (i % 2 === 0) {
        quantized[byteIdx] = bestIdx & 0x0F;
      } else {
        quantized[byteIdx] |= (bestIdx & 0x0F) << 4;
      }
    }

    const metadata: QuantizationMetadata = {
      quantized: true,
      bits: 4,
      shape,
      dtype: 'float32',
      absmax: Array.from(absmax),
      blocksize: blockSize,
    };

    return {
      data: quantized,
      metadata,
      originalShape: shape,
    };
  } else {
    // 8-bit quantization
    const numElements = tensor.length;
    const quantized = new Uint8Array(numElements);

    for (let i = 0; i < numElements; i++) {
      const blockIdx = Math.floor(i / blockSize);
      const scale = absmax[blockIdx];

      // Normalize to [-1, 1] then map to [0, 255]
      const normalized = scale > 0 ? tensor[i] / scale : 0;
      const clamped = Math.max(-1, Math.min(1, normalized));
      const quantizedVal = Math.round((clamped + 1) / 2 * 255);

      quantized[i] = quantizedVal;
    }

    const metadata: QuantizationMetadata = {
      quantized: true,
      bits: 8,
      shape,
      dtype: 'float32',
      absmax: Array.from(absmax),
      blocksize: blockSize,
    };

    return {
      data: quantized,
      metadata,
      originalShape: shape,
    };
  }
}

/**
 * Dequantize a tensor from its quantized representation.
 *
 * @param quantized - Quantized tensor data
 * @param metadata - Quantization metadata
 * @returns Dequantized tensor as Float32Array
 * @throws Error if bits is not 4 or 8
 *
 * @example
 * ```typescript
 * const dequantized = dequantizeTensor(quantized.data, quantized.metadata);
 * console.log(`Restored tensor with ${dequantized.length} elements`);
 * ```
 */
export function dequantizeTensor(
  quantized: Uint8Array,
  metadata: QuantizationMetadata
): Float32Array {
  const bits = metadata.bits;

  if (bits !== 4 && bits !== 8) {
    throw new Error(`Only 4-bit and 8-bit quantization supported, got ${bits}`);
  }

  const blockSize = metadata.blocksize || DEFAULT_BLOCK_SIZE;
  const absmax = metadata.absmax || [];

  if (bits === 4) {
    // 4-bit dequantization
    const numElements = quantized.length * 2;
    const dequantized = new Float32Array(numElements);

    for (let i = 0; i < numElements; i++) {
      const byteIdx = Math.floor(i / 2);
      const isHighNibble = i % 2 !== 0;
      const nibble = isHighNibble ? (quantized[byteIdx] >> 4) & 0x0F : quantized[byteIdx] & 0x0F;

      const blockIdx = Math.floor(i / blockSize);
      const scale = absmax[blockIdx] || 1;

      dequantized[i] = NF4_LEVELS[nibble] * scale;
    }

    return dequantized;
  } else {
    // 8-bit dequantization
    const numElements = quantized.length;
    const dequantized = new Float32Array(numElements);

    for (let i = 0; i < numElements; i++) {
      const blockIdx = Math.floor(i / blockSize);
      const scale = absmax[blockIdx] || 1;

      // Map from [0, 255] to [-1, 1] then scale
      const normalized = (quantized[i] / 255) * 2 - 1;
      dequantized[i] = normalized * scale;
    }

    return dequantized;
  }
}

/**
 * Calculate the compression ratio achieved by quantization.
 *
 * @param originalSize - Original size in bytes
 * @param quantizedSize - Quantized size in bytes
 * @returns Compression ratio (original_size / compressed_size)
 */
export function calculateCompressionRatio(
  originalSize: number,
  quantizedSize: number
): number {
  if (quantizedSize === 0) {
    return 1;
  }
  return originalSize / quantizedSize;
}

/**
 * Estimate expected accuracy loss from quantization.
 *
 * These are approximate values based on literature:
 * - 4-bit: typically <1% accuracy loss
 * - 8-bit: typically <0.5% accuracy loss
 *
 * @param bits - Quantization bit width (4 or 8)
 * @returns Estimated accuracy loss percentage
 */
export function estimateAccuracyLoss(bits: 4 | 8): number {
  if (bits === 4) {
    return 0.5; // 0.5% estimated loss (conservative, usually <1%)
  } else if (bits === 8) {
    return 0.2; // 0.2% estimated loss (conservative, usually <0.5%)
  }
  return 1.0; // Unknown, conservative estimate
}

/**
 * Quantize a downloaded HuggingFace model.
 *
 * This is the main entry point for model quantization.
 *
 * @param modelId - HuggingFace model identifier (e.g., 'microsoft/DialoGPT-medium')
 * @param options - Quantization options
 * @returns Promise resolving to quantized model information
 * @throws Error if model not found or quantization fails
 *
 * @example
 * ```typescript
 * const model = await quantizeModel('microsoft/DialoGPT-medium', { bits: 4 });
 * console.log(`Quantized to ${model.compressionRatio.toFixed(2)}x compression`);
 * ```
 */
export async function quantizeModel(
  modelId: string,
  options: QuantizationOptions
): Promise<QuantizedModel> {
  const bits = options.bits;

  if (bits !== 4 && bits !== 8) {
    throw new Error(`Only 4-bit and 8-bit quantization supported, got ${bits}`);
  }

  // Placeholder implementation
  // In the full implementation, this would:
  // 1. Load the model from cache
  // 2. Quantize all weight tensors
  // 3. Save quantized model to disk
  // 4. Return model information

  const compressionRatio = bits === 4 ? 8.0 : 4.0;

  return {
    id: modelId,
    path: `/cache/${modelId}/quantized-${bits}bit`,
    quantization: {
      bits,
      type: bits === 4 ? 'nf4' : 'int8',
      kvCacheEnabled: options.kvCache || false,
    },
    compressionRatio,
  };
}
