/**
 * Quantization module for weight and KV cache compression.
 *
 * This module provides:
 * - Weight quantization (4-bit, 8-bit block-wise)
 * - KV cache quantization (3-bit keys, 2-bit values via TurboQuant)
 *
 * @module quantization
 */

// Export types
export type {
  QuantizationOptions,
  QuantizedModel,
  QuantizationMetadata,
  QuantizedTensor,
} from './types.js';

// Export weight quantization
export {
  quantizeTensor,
  dequantizeTensor,
  calculateCompressionRatio,
  estimateAccuracyLoss,
  quantizeModel,
} from './weight.js';

// Export KV cache quantization
export {
  computeCosineSimilarity,
  estimateCompressionRatio,
  LloydMaxQuantizer,
  OrthogonalRotation,
  QJLProjection,
  TurboQuantKeyCompressor,
  GroupValueQuantizer,
  KVCacheQuantizer,
} from './kv_cache.js';
