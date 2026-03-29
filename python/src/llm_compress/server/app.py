"""FastAPI application factory.

This module creates the OpenAI-compatible API server.
"""


from fastapi import FastAPI


def create_app(model_id: str, backend: str = "vllm") -> FastAPI:
    """Create FastAPI application.
    
    Args:
        model_id: HuggingFace model identifier
        backend: Inference backend ("vllm" or "llama-cpp")
        
    Returns:
        Configured FastAPI application
        
    Note:
        This is a placeholder. Full implementation in api-server.
    """
    app = FastAPI(
        title="LLM Compress API",
        description="OpenAI-compatible API for quantized LLMs",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    return app
