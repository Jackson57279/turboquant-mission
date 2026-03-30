/**
 * LLM Compress: Unified LLM Quantization & Inference System
 * TypeScript Implementation
 *
 * This package provides:
 * - Weight quantization (4-bit, 8-bit block-wise)
 * - KV cache compression (3-bit keys, 2-bit values via TurboQuant)
 * - Layer-wise loading for low-memory inference
 * - OpenAI-compatible API server
 *
 * @example
 * ```typescript
 * import { quantizeModel, getBackend } from 'llm-compress';
 *
 * const model = await quantizeModel('meta-llama/Llama-2-7b', { bits: 4 });
 * console.log(`Quantized model: ${model.id}`);
 * ```
 *
 * @packageDocumentation
 */

export const VERSION = '0.1.0';
export const AUTHOR = 'llm-compress Team';
export const LICENSE = 'MIT';

// Export types
export type { QuantizationOptions, QuantizedModel } from './quantization/types.js';
export type { Backend, BackendConfig } from './backends/types.js';

// Export main functions
export { quantizeTensor, quantizeModel } from './quantization/weight.js';
export { getBackend, listBackends } from './backends/index.js';
