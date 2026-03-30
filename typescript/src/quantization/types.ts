/**
 * Type definitions for quantization module.
 *
 * @module quantization/types
 */

/**
 * Options for model quantization.
 */
export interface QuantizationOptions {
  /** Quantization bit width (4 or 8) */
  bits: 4 | 8;
  /** Whether to enable KV cache quantization */
  kvCache?: boolean;
  /** Custom cache directory */
  cacheDir?: string;
  /** Whether to enable layer-wise loading */
  enableLayerWise?: boolean;
}

/**
 * Represents a quantized model.
 */
export interface QuantizedModel {
  /** Model identifier */
  id: string;
  /** Path to quantized model directory */
  path: string;
  /** Quantization configuration */
  quantization: {
    bits: 4 | 8;
    type: 'nf4' | 'int8';
    kvCacheEnabled: boolean;
  };
  /** Compression ratio achieved */
  compressionRatio: number;
}

/**
 * Quantization metadata for a tensor.
 */
export interface QuantizationMetadata {
  /** Whether this tensor was quantized */
  quantized: boolean;
  /** Bit width (4 or 8) */
  bits?: number;
  /** Original tensor shape */
  shape?: number[];
  /** Original tensor dtype */
  dtype?: string;
  /** Quantization absmax values */
  absmax?: number[];
  /** Block size for quantization */
  blocksize?: number;
  /** Error message if quantization failed */
  error?: string;
  /** Reason for exclusion if not quantized */
  reason?: string;
}

/**
 * Quantized tensor data.
 */
export interface QuantizedTensor {
  /** Quantized tensor data */
  data: Uint8Array;
  /** Quantization metadata */
  metadata: QuantizationMetadata;
  /** Original shape */
  originalShape: number[];
}
