"""OpenAI-compatible API models.

This module defines Pydantic models for request/response validation,
following the OpenAI API specification for chat completions and models.

References:
    - OpenAI API: https://platform.openai.com/docs/api-reference
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Chat Completions Models
# ============================================================================


class ChatMessage(BaseModel):
    """A single chat message.

    Attributes:
        role: The role of the message author (system, user, assistant)
        content: The content of the message
        name: Optional name for the participant
    """
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="The role of the message author"
    )
    content: str = Field(
        ...,
        description="The content of the message"
    )
    name: str | None = Field(
        None,
        description="The name of the author"
    )


class ChatCompletionRequest(BaseModel):
    """Chat completion request model.

    Attributes:
        model: ID of the model to use
        messages: List of messages comprising the conversation
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter (0.0-1.0)
        stop: Stop sequences to halt generation
        stream: Whether to stream responses
        presence_penalty: Presence penalty (-2.0 to 2.0)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
    """
    model: str = Field(
        ...,
        description="ID of the model to use"
    )
    messages: list[ChatMessage] = Field(
        ...,
        description="A list of messages comprising the conversation",
        min_length=1
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        256,
        ge=1,
        description="Maximum number of tokens to generate"
    )
    top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability"
    )
    stop: str | list[str] | None = Field(
        None,
        description="Stop sequences"
    )
    stream: bool = Field(
        False,
        description="Whether to stream back partial progress"
    )
    presence_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )

    @field_validator("stop", mode="before")
    @classmethod
    def validate_stop(cls, v: Any) -> list[str] | None:
        """Normalize stop to list of strings."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError("stop must be a string or list of strings")


class ChatCompletionChoice(BaseModel):
    """A single chat completion choice.

    Attributes:
        index: Index of this choice
        message: The generated message
        finish_reason: Reason for stopping generation
    """
    index: int = Field(..., description="The index of this choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Literal["stop", "length", "content_filter"] | None = Field(
        None,
        description="The reason the completion finished"
    )


class ChatCompletionChunkChoice(BaseModel):
    """A streaming chat completion chunk choice.

    Attributes:
        index: Index of this choice
        delta: The delta content for this chunk
        finish_reason: Reason for stopping generation
    """
    index: int = Field(..., description="The index of this choice")
    delta: dict[str, Any] = Field(
        ...,
        description="The delta content for this chunk"
    )
    finish_reason: Literal["stop", "length", "content_filter"] | None = Field(
        None,
        description="The reason the completion finished"
    )


class UsageInfo(BaseModel):
    """Token usage information.

    Attributes:
        prompt_tokens: Tokens in the prompt
        completion_tokens: Tokens in the completion
        total_tokens: Total tokens used
    """
    prompt_tokens: int = Field(..., description="Tokens in the prompt")
    completion_tokens: int = Field(..., description="Tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """Chat completion response model.

    Attributes:
        id: Unique identifier for this completion
        object: Object type (always "chat.completion")
        created: Unix timestamp
        model: The model used
        choices: List of completion choices
        usage: Token usage information
    """
    id: str = Field(..., description="A unique identifier for the completion")
    object: Literal["chat.completion"] = Field(
        "chat.completion",
        description="The object type"
    )
    created: int = Field(..., description="Unix timestamp when created")
    model: str = Field(..., description="The model used for completion")
    choices: list[ChatCompletionChoice] = Field(
        ...,
        description="A list of completion choices"
    )
    usage: UsageInfo = Field(..., description="Token usage information")


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk model.

    Attributes:
        id: Unique identifier for this completion
        object: Object type (always "chat.completion.chunk")
        created: Unix timestamp
        model: The model used
        choices: List of chunk choices
    """
    id: str = Field(..., description="A unique identifier for the completion")
    object: Literal["chat.completion.chunk"] = Field(
        "chat.completion.chunk",
        description="The object type"
    )
    created: int = Field(..., description="Unix timestamp when created")
    model: str = Field(..., description="The model used for completion")
    choices: list[ChatCompletionChunkChoice] = Field(
        ...,
        description="A list of chunk choices"
    )


# ============================================================================
# Legacy Completions Models
# ============================================================================


class CompletionRequest(BaseModel):
    """Legacy text completion request model.

    Attributes:
        model: ID of the model to use
        prompt: The prompt(s) to generate completions for
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter (0.0-1.0)
        stop: Stop sequences to halt generation
        stream: Whether to stream responses
    """
    model: str = Field(
        ...,
        description="ID of the model to use"
    )
    prompt: str | list[str] = Field(
        ...,
        description="The prompt(s) to generate completions for"
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        256,
        ge=1,
        description="Maximum number of tokens to generate"
    )
    top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability"
    )
    stop: str | list[str] | None = Field(
        None,
        description="Stop sequences"
    )
    stream: bool = Field(
        False,
        description="Whether to stream back partial progress"
    )

    @field_validator("stop", mode="before")
    @classmethod
    def validate_stop(cls, v: Any) -> list[str] | None:
        """Normalize stop to list of strings."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError("stop must be a string or list of strings")

    @field_validator("prompt", mode="before")
    @classmethod
    def validate_prompt(cls, v: Any) -> str | list[str]:
        """Normalize prompt to string or list of strings."""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return v
        raise ValueError("prompt must be a string or list of strings")


class CompletionChoice(BaseModel):
    """A single text completion choice.

    Attributes:
        index: Index of this choice
        text: The generated text
        finish_reason: Reason for stopping generation
        logprobs: Log probabilities (if requested)
    """
    index: int = Field(..., description="The index of this choice")
    text: str = Field(..., description="The generated text")
    finish_reason: Literal["stop", "length", "content_filter"] | None = Field(
        None,
        description="The reason the completion finished"
    )
    logprobs: dict[str, Any] | None = Field(
        None,
        description="Log probabilities (if requested)"
    )


class CompletionChunkChoice(BaseModel):
    """A streaming text completion chunk choice.

    Attributes:
        index: Index of this choice
        text: The generated text chunk
        finish_reason: Reason for stopping generation
    """
    index: int = Field(..., description="The index of this choice")
    text: str = Field(..., description="The generated text chunk")
    finish_reason: Literal["stop", "length", "content_filter"] | None = Field(
        None,
        description="The reason the completion finished"
    )


class CompletionResponse(BaseModel):
    """Text completion response model.

    Attributes:
        id: Unique identifier for this completion
        object: Object type (always "text_completion")
        created: Unix timestamp
        model: The model used
        choices: List of completion choices
        usage: Token usage information
    """
    id: str = Field(..., description="A unique identifier for the completion")
    object: Literal["text_completion"] = Field(
        "text_completion",
        description="The object type"
    )
    created: int = Field(..., description="Unix timestamp when created")
    model: str = Field(..., description="The model used for completion")
    choices: list[CompletionChoice] = Field(
        ...,
        description="A list of completion choices"
    )
    usage: UsageInfo = Field(..., description="Token usage information")


class CompletionChunk(BaseModel):
    """Streaming text completion chunk model.

    Attributes:
        id: Unique identifier for this completion
        object: Object type (always "text_completion.chunk")
        created: Unix timestamp
        model: The model used
        choices: List of chunk choices
    """
    id: str = Field(..., description="A unique identifier for the completion")
    object: Literal["text_completion.chunk"] = Field(
        "text_completion.chunk",
        description="The object type"
    )
    created: int = Field(..., description="Unix timestamp when created")
    model: str = Field(..., description="The model used for completion")
    choices: list[CompletionChunkChoice] = Field(
        ...,
        description="A list of chunk choices"
    )


# ============================================================================
# Models Endpoint Models
# ============================================================================


class ModelInfo(BaseModel):
    """Information about a model.

    Attributes:
        id: Model identifier
        object: Object type (always "model")
        created: Unix timestamp
        owned_by: Organization that owns the model
    """
    id: str = Field(..., description="The model identifier")
    object: Literal["model"] = Field("model", description="The object type")
    created: int = Field(..., description="Unix timestamp when created")
    owned_by: str = Field(..., description="The organization that owns the model")


class ModelListResponse(BaseModel):
    """Response from the models endpoint.

    Attributes:
        object: Object type (always "list")
        data: List of available models
    """
    object: Literal["list"] = Field("list", description="The object type")
    data: list[ModelInfo] = Field(..., description="A list of available models")


# ============================================================================
# Health & Error Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Server status (healthy or unhealthy)
        model: Model being served
        backend: Backend being used
        version: API version
    """
    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="The server status"
    )
    model: str | None = Field(None, description="The model being served")
    backend: str | None = Field(None, description="The backend being used")
    version: str = Field("0.1.0", description="The API version")


class ErrorDetail(BaseModel):
    """Error detail model.

    Attributes:
        message: Error message
        type: Error type
        param: Parameter that caused the error (if applicable)
        code: Error code (if applicable)
    """
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: str | None = Field(None, description="Parameter that caused the error")
    code: str | None = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response model.

    Attributes:
        error: Error details
    """
    error: ErrorDetail = Field(..., description="Error details")
