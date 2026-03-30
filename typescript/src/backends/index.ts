/**
 * Backend module for pluggable inference backends.
 *
 * This module provides adapters for different inference backends:
 * - vLLM: High-throughput GPU inference
 * - llama.cpp: Broad hardware support including CPU
 *
 * @module backends
 */

import type { Backend, BackendConfig, BackendType } from './types.js';

/** Registry of available backends */
const backendRegistry: Map<BackendType, new (config: BackendConfig) => Backend> = new Map();

/**
 * Register a backend implementation.
 *
 * @param type - Backend type identifier
 * @param implementation - Backend class constructor
 */
export function registerBackend(
  type: BackendType,
  implementation: new (config: BackendConfig) => Backend
): void {
  backendRegistry.set(type, implementation);
}

/**
 * Get a backend instance by type.
 *
 * @param type - Backend type ('vllm' or 'llama-cpp')
 * @param config - Backend configuration
 * @returns Backend instance
 * @throws Error if backend type is not registered
 *
 * @example
 * ```typescript
 * const backend = getBackend('vllm', { modelId: 'meta-llama/Llama-2-7b' });
 * await backend.initialize();
 * ```
 */
export function getBackend(type: BackendType, config: BackendConfig): Backend {
  const BackendClass = backendRegistry.get(type);

  if (!BackendClass) {
    throw new Error(`Backend '${type}' not found. Available backends: ${listBackends().join(', ')}`);
  }

  return new BackendClass(config);
}

/**
 * List all available backend types.
 *
 * @returns Array of registered backend type names
 */
export function listBackends(): string[] {
  return Array.from(backendRegistry.keys());
}

/**
 * Check if a backend type is available.
 *
 * @param type - Backend type to check
 * @returns True if the backend is registered
 */
export function isBackendAvailable(type: string): boolean {
  return backendRegistry.has(type as BackendType);
}

/**
 * Base backend class that all backends extend.
 */
export abstract class BaseBackend implements Backend {
  /** Backend configuration */
  readonly config: BackendConfig;
  protected initialized = false;

  constructor(config: BackendConfig) {
    this.config = config;
  }

  /**
   * Check if the backend is initialized.
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Initialize the backend.
   */
  abstract initialize(): Promise<void>;

  /**
   * Get backend health status.
   */
  abstract health(): Promise<{ status: string; details?: Record<string, unknown> }>;

  /**
   * Generate a chat completion.
   */
  abstract chatCompletion(request: import('./types.js').ChatCompletionRequest): Promise<import('./types.js').ChatCompletionResponse>;

  /**
   * Dispose of backend resources.
   */
  abstract dispose(): Promise<void>;
}

// Export types
export type { Backend, BackendConfig, BackendType } from './types.js';
