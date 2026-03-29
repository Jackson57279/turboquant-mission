"""FastAPI application factory.

This module creates the OpenAI-compatible API server with support for:
- GET /health - Health check endpoint
- GET /v1/models - List available models
- GET /v1/models/{model_id} - Get model information
- POST /v1/chat/completions - Chat completions (streaming and non-streaming)
- POST /v1/completions - Legacy text completions (streaming and non-streaming)

References:
    - OpenAI API: https://platform.openai.com/docs/api-reference
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from llm_compress.backends.registry import get_backend

from .models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)

# Global state for the server
_server_state: dict[str, Any] = {
    "backend": None,
    "model_id": None,
    "backend_name": None,
    "cache_dir": None,
}


def _generate_id(prefix: str = "llmc") -> str:
    """Generate a unique ID for completions."""
    import uuid
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _convert_messages_to_list(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert Pydantic ChatMessage objects to dict format for backend."""
    return [{"role": msg.role, "content": msg.content} for msg in messages]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup: Nothing special needed, backend is initialized lazily
    logger.info("API server starting up")
    yield
    # Shutdown: Clean up backend
    logger.info("API server shutting down")
    if _server_state["backend"] is not None:
        try:
            _server_state["backend"].shutdown()
        except Exception as e:
            logger.error(f"Error shutting down backend: {e}")


def create_app(
    model_id: str,
    backend: str = "vllm",
    cache_dir: str | None = None,
    enable_kv_compression: bool = True,
) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        model_id: HuggingFace model identifier
        backend: Inference backend ("vllm" or "llama-cpp")
        cache_dir: Cache directory for model files
        enable_kv_compression: Whether to enable TurboQuant KV cache compression
        
    Returns:
        Configured FastAPI application
    """
    # Store server state
    _server_state["model_id"] = model_id
    _server_state["backend_name"] = backend
    _server_state["cache_dir"] = cache_dir
    _server_state["enable_kv_compression"] = enable_kv_compression

    app = FastAPI(
        title="LLM Compress API",
        description="OpenAI-compatible API for quantized LLMs",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_backend() -> Any:
        """Get or initialize the backend."""
        if _server_state["backend"] is None:
            logger.info(f"Initializing {backend} backend with model {model_id}")
            try:
                backend_instance = get_backend(
                    name=backend,
                    model_id=model_id,
                    enable_kv_compression=enable_kv_compression,
                )
                backend_instance.initialize()
                _server_state["backend"] = backend_instance
                logger.info("Backend initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize backend: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Backend initialization failed: {str(e)}"
                )
        return _server_state["backend"]

    # ========================================================================
    # Health Endpoint
    # ========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        if _server_state["backend"] is None:
            # Backend not initialized yet - we're healthy but not ready
            return HealthResponse(
                status="healthy",
                model=model_id,
                backend=backend,
                version="0.1.0",
            )

        try:
            health_info = _server_state["backend"].health()
            return HealthResponse(
                status="healthy" if health_info.get("status") == "healthy" else "unhealthy",
                model=model_id,
                backend=backend,
                version="0.1.0",
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                model=model_id,
                backend=backend,
                version="0.1.0",
            )

    # ========================================================================
    # Models Endpoints
    # ========================================================================

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        """List available models."""
        # For now, we only serve the configured model
        model_info = ModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="llm-compress",
        )
        return ModelListResponse(data=[model_info])

    @app.get("/v1/models/{model_id:path}", response_model=ModelInfo)
    async def get_model(model_id: str) -> ModelInfo:
        """Get information about a specific model."""
        # Check if the requested model matches our served model
        if model_id != _server_state["model_id"]:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found. Available: {_server_state['model_id']}"
            )

        return ModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="llm-compress",
        )

    # ========================================================================
    # Chat Completions Endpoint
    # ========================================================================

    async def _stream_chat_completion(
        backend: Any,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks."""
        messages = _convert_messages_to_list(request.messages)

        try:
            # Get streaming response from backend
            stream = backend.chat(
                messages=messages,
                max_tokens=request.max_tokens or 256,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=True,
            )

            completion_id = _generate_id("chat")
            created = int(time.time())

            for chunk in stream:
                # Convert backend chunk to OpenAI format
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

                chat_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=delta,
                            finish_reason=finish_reason,
                        )
                    ],
                )

                # Yield as SSE
                yield f"data: {chat_chunk.model_dump_json()}\n\n"

            # End stream
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming chat completion error: {e}")
            error_chunk = ChatCompletionChunk(
                id=_generate_id("chat"),
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta={"role": "assistant", "content": f"Error: {str(e)}"},
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        """Create a chat completion.
        
        Supports both streaming and non-streaming responses.
        """
        backend = _get_backend()

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                _stream_chat_completion(backend, request),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            messages = _convert_messages_to_list(request.messages)

            try:
                result = backend.chat(
                    messages=messages,
                    max_tokens=request.max_tokens or 256,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    stream=False,
                )

                # Convert backend result to OpenAI format
                created = int(time.time())
                choices = []

                for i, choice in enumerate(result.get("choices", [])):
                    message = ChatMessage(
                        role="assistant",
                        content=choice.get("message", {}).get("content", ""),
                    )
                    choices.append(
                        ChatCompletionChoice(
                            index=i,
                            message=message,
                            finish_reason=choice.get("finish_reason"),
                        )
                    )

                usage_info = result.get("usage", {})
                usage = UsageInfo(
                    prompt_tokens=usage_info.get("prompt_tokens", 0),
                    completion_tokens=usage_info.get("completion_tokens", 0),
                    total_tokens=usage_info.get("total_tokens", 0),
                )

                return ChatCompletionResponse(
                    id=result.get("id", _generate_id("chat")),
                    created=created,
                    model=request.model,
                    choices=choices,
                    usage=usage,
                )

            except Exception as e:
                logger.error(f"Chat completion error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Completion failed: {str(e)}"
                )

    # ========================================================================
    # Legacy Completions Endpoint
    # ========================================================================

    async def _stream_completion(
        backend: Any,
        request: CompletionRequest,
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """Stream completion chunks."""
        try:
            # Get streaming response from backend
            stream = backend.generate(
                prompt=prompt,
                max_tokens=request.max_tokens or 256,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=True,
            )

            completion_id = _generate_id("cmpl")
            created = int(time.time())

            for chunk in stream:
                # Convert backend chunk to OpenAI format
                text = chunk.get("choices", [{}])[0].get("text", "")
                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

                completion_chunk = CompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        CompletionChunkChoice(
                            index=0,
                            text=text,
                            finish_reason=finish_reason,
                        )
                    ],
                )

                # Yield as SSE
                yield f"data: {completion_chunk.model_dump_json()}\n\n"

            # End stream
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming completion error: {e}")
            error_chunk = CompletionChunk(
                id=_generate_id("cmpl"),
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChunkChoice(
                        index=0,
                        text=f"Error: {str(e)}",
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> Any:
        """Create a text completion (legacy endpoint).
        
        Supports both streaming and non-streaming responses.
        """
        backend = _get_backend()

        # Handle list of prompts - for simplicity, just use first prompt
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                _stream_completion(backend, request, prompt),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            try:
                result = backend.generate(
                    prompt=prompt,
                    max_tokens=request.max_tokens or 256,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    stream=False,
                )

                # Convert backend result to OpenAI format
                created = int(time.time())
                choices = []

                for i, choice in enumerate(result.get("choices", [])):
                    choices.append(
                        CompletionChoice(
                            index=i,
                            text=choice.get("text", ""),
                            finish_reason=choice.get("finish_reason"),
                        )
                    )

                usage_info = result.get("usage", {})
                usage = UsageInfo(
                    prompt_tokens=usage_info.get("prompt_tokens", 0),
                    completion_tokens=usage_info.get("completion_tokens", 0),
                    total_tokens=usage_info.get("total_tokens", 0),
                )

                return CompletionResponse(
                    id=result.get("id", _generate_id("cmpl")),
                    created=created,
                    model=request.model,
                    choices=choices,
                    usage=usage,
                )

            except Exception as e:
                logger.error(f"Completion error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Completion failed: {str(e)}"
                )

    # ========================================================================
    # Error Handlers
    # ========================================================================

    from fastapi.responses import JSONResponse

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> Any:
        """Handle HTTP exceptions."""
        error_response = ErrorResponse(
            error={
                "message": exc.detail,
                "type": "invalid_request_error" if exc.status_code < 500 else "api_error",
                "param": None,
                "code": str(exc.status_code),
            }
        )
        return JSONResponse(
            content=error_response.model_dump(),
            status_code=exc.status_code,
        )

    return app
