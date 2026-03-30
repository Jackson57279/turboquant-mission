/**
 * Type definitions for backend module.
 *
 * @module backends/types
 */

/** Supported backend types */
export type BackendType = 'vllm' | 'llama-cpp';

/**
 * Configuration for a backend.
 */
export interface BackendConfig {
  /** Model identifier */
  modelId: string;
  /** Path to model directory */
  modelPath?: string;
  /** Whether to use GPU acceleration */
  useGpu?: boolean;
  /** GPU device ID (if applicable) */
  deviceId?: number;
  /** Maximum context length */
  maxContextLength?: number;
  /** Batch size for inference */
  batchSize?: number;
  /** Additional backend-specific options */
  options?: Record<string, unknown>;
}

/**
 * Chat message for API requests.
 */
export interface ChatMessage {
  /** Message role ('system', 'user', or 'assistant') */
  role: 'system' | 'user' | 'assistant';
  /** Message content */
  content: string;
}

/**
 * Chat completion request.
 */
export interface ChatCompletionRequest {
  /** Model identifier */
  model: string;
  /** Array of chat messages */
  messages: ChatMessage[];
  /** Whether to stream the response */
  stream?: boolean;
  /** Sampling temperature (0.0-2.0) */
  temperature?: number;
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Stop sequences */
  stop?: string[];
  /** Top-p sampling */
  topP?: number;
}

/**
 * Chat completion response.
 */
export interface ChatCompletionResponse {
  /** Unique identifier for the completion */
  id: string;
  /** Object type */
  object: string;
  /** Creation timestamp */
  created: number;
  /** Model identifier */
  model: string;
  /** Completion choices */
  choices: Array<{
    /** Choice index */
    index: number;
    /** Generated message */
    message: ChatMessage;
    /** Finish reason */
    finishReason: string;
  }>;
  /** Usage statistics */
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

/**
 * Backend interface that all backend implementations must implement.
 */
export interface Backend {
  /** Backend configuration */
  readonly config: BackendConfig;

  /**
   * Check if the backend is initialized.
   * @returns True if initialized
   */
  isInitialized(): boolean;

  /**
   * Initialize the backend with the configured model.
   */
  initialize(): Promise<void>;

  /**
   * Get backend health status.
   * @returns Health status object
   */
  health(): Promise<{ status: string; details?: Record<string, unknown> }>;

  /**
   * Generate a chat completion.
   * @param request - Completion request
   * @returns Completion response
   */
  chatCompletion(request: ChatCompletionRequest): Promise<ChatCompletionResponse>;

  /**
   * Dispose of backend resources.
   */
  dispose(): Promise<void>;
}
