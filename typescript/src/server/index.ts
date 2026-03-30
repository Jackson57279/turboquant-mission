/**
 * OpenAI-compatible API server.
 *
 * This module provides a FastAPI-style HTTP server for serving
 * quantized models with an OpenAI-compatible API.
 *
 * @module server/app
 */

import type { ModelInfo, ListModelsResponse } from './types.js';

/**
 * Health check response.
 */
export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
}

/**
 * Create a health response.
 *
 * @returns Health check response
 */
export function createHealthResponse(): HealthResponse {
  return {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '0.1.0',
  };
}

/**
 * List available models.
 *
 * @returns List of available models
 */
export function listModels(): ListModelsResponse {
  return {
    object: 'list',
    data: [],
  };
}

/**
 * Create a model info object.
 *
 * @param modelId - Model identifier
 * @returns Model info
 */
export function createModelInfo(modelId: string): ModelInfo {
  return {
    id: modelId,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'llm-compress',
  };
}
