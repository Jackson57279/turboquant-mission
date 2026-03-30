/**
 * Type definitions for server module.
 *
 * @module server/types
 */

/**
 * Model information response.
 */
export interface ModelInfo {
  /** Model identifier */
  id: string;
  /** Object type */
  object: string;
  /** Creation timestamp */
  created: number;
  /** Owner */
  owned_by: string;
  /** Quantization info */
  quantization?: {
    bits: number;
    type: string;
  };
}

/**
 * List models response.
 */
export interface ListModelsResponse {
  /** Object type */
  object: string;
  /** Available models */
  data: ModelInfo[];
}
