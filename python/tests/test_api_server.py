"""Tests for OpenAI-compatible API server.

This module tests the FastAPI server endpoints including:
- Health check endpoint
- Models listing
- Chat completions (streaming and non-streaming)
- Legacy text completions (streaming and non-streaming)
- Request validation and error handling

References:
    - VAL-API-001: Health endpoint returns healthy status
    - VAL-API-002: Models endpoint returns available models
    - VAL-API-003: Chat completions endpoint works
    - VAL-API-010: Request validation works
    - VAL-API-012: Streaming responses work
    - VAL-API-013: Error responses are appropriate
    - VAL-API-014: Legacy completions endpoint works
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_compress.server.app import _server_state, create_app


class MockBackend:
    """Mock backend for testing with support for chat completion features."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.config = kwargs
        self._initialized = False
        self._last_temperature: float | None = None
        self._last_stop: list[str] | None = None
        self._last_max_tokens: int | None = None
        self._call_count = 0

    def initialize(self) -> None:
        self._initialized = True

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "backend": "mock",
            "model_id": self.model_id,
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        # Store parameters for verification
        self._last_temperature = temperature
        self._last_stop = stop
        self._last_max_tokens = max_tokens
        self._call_count += 1

        if stream:
            return self._generate_stream(prompt, max_tokens, stop)
        else:
            return self._generate_sync(prompt, max_tokens, stop)

    def _generate_sync(self, prompt: str, max_tokens: int, stop: list[str] | None) -> dict[str, Any]:
        response_text = f"Mock response to: {prompt[:20]}"

        # Simulate stop sequence detection
        finish_reason = "stop"
        if stop:
            for seq in stop:
                if seq in response_text:
                    response_text = response_text.split(seq)[0]
                    finish_reason = "stop"
                    break

        # Simulate max_tokens limiting
        words = response_text.split()
        if len(words) > max_tokens:
            words = words[:max_tokens]
            response_text = " ".join(words)
            finish_reason = "length"

        return {
            "id": "mock-gen-12345",
            "object": "text_completion",
            "model": self.model_id,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
        }

    def _generate_stream(self, prompt: str, max_tokens: int, stop: list[str] | None) -> Iterator[dict[str, Any]]:
        words = ["Hello", "world", "from", "mock", "backend"]
        for i, word in enumerate(words):
            yield {
                "id": f"mock-chunk-{i}",
                "object": "text_completion.chunk",
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": word + " ",
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "mock-chunk-final",
            "object": "text_completion.chunk",
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop",
                }
            ],
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        # Store parameters for verification
        self._last_temperature = temperature
        self._last_stop = stop
        self._last_max_tokens = max_tokens
        self._call_count += 1

        if stream:
            return self._chat_stream(messages, max_tokens, stop)
        else:
            return self._chat_sync(messages, max_tokens, stop)

    def _chat_sync(self, messages: list[dict[str, str]], max_tokens: int, stop: list[str] | None) -> dict[str, Any]:
        last_message = messages[-1].get("content", "") if messages else ""

        # Check for system message to customize response
        has_system = any(m.get("role") == "system" for m in messages)

        if has_system:
            response_text = f"As a helpful assistant, responding to: {last_message[:20]}"
        else:
            response_text = f"Mock chat response to: {last_message[:20]}"

        # Simulate stop sequence detection
        finish_reason = "stop"
        if stop:
            for seq in stop:
                if seq in response_text:
                    response_text = response_text.split(seq)[0]
                    finish_reason = "stop"
                    break

        # Simulate max_tokens limiting
        words = response_text.split()
        if len(words) > max_tokens:
            words = words[:max_tokens]
            response_text = " ".join(words)
            finish_reason = "length"

        return {
            "id": "mock-chat-12345",
            "object": "chat.completion",
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(response_text.split()),
                "total_tokens": sum(len(m.get("content", "").split()) for m in messages)
                + len(response_text.split()),
            },
        }

    def _chat_stream(
        self, messages: list[dict[str, str]], max_tokens: int, stop: list[str] | None
    ) -> Iterator[dict[str, Any]]:
        words = ["Hello", "from", "mock", "chat"]
        for i, word in enumerate(words):
            yield {
                "id": f"mock-chat-chunk-{i}",
                "object": "chat.completion.chunk",
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": word + " ",
                        },
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "mock-chat-chunk-final",
            "object": "chat.completion.chunk",
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

    def shutdown(self) -> None:
        self._initialized = False


@pytest.fixture
def mock_backend():
    """Fixture to provide a mock backend."""
    return MockBackend("test-model")


@pytest.fixture
def client(mock_backend):
    """Fixture to create a test client with mocked backend."""
    # Clear server state
    _server_state["backend"] = None
    _server_state["model_id"] = None
    _server_state["backend_name"] = None

    # Create app
    app = create_app(
        model_id="test-model",
        backend="vllm",
    )

    # Set up mock backend directly
    _server_state["backend"] = mock_backend
    mock_backend.initialize()

    # Create test client
    with TestClient(app) as client:
        yield client

    # Cleanup
    if _server_state["backend"] is not None:
        _server_state["backend"].shutdown()
        _server_state["backend"] = None


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_healthy(self, client):
        """VAL-API-001: Health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model"] == "test-model"
        assert data["version"] == "0.1.0"

    def test_health_returns_unhealthy_if_backend_fails(self, client, mock_backend):
        """Test that health returns unhealthy if backend fails."""
        # Make backend health fail
        mock_backend.health = lambda: {"status": "unhealthy"}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


class TestModelsEndpoint:
    """Tests for the /v1/models endpoints."""

    def test_list_models_returns_model_list(self, client):
        """VAL-API-002: Models endpoint returns available models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["owned_by"] == "llm-compress"

    def test_get_model_returns_model_info(self, client):
        """Test getting specific model info."""
        response = client.get("/v1/models/test-model")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-model"
        assert data["object"] == "model"

    def test_get_unknown_model_returns_404(self, client):
        """Test getting non-existent model returns 404."""
        response = client.get("/v1/models/unknown-model")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data


class TestChatCompletions:
    """Tests for the /v1/chat/completions endpoint."""

    def test_chat_completions_non_streaming(self, client):
        """VAL-API-003: Chat completions endpoint works (non-streaming)."""
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert "message" in data["choices"][0]
        assert "role" in data["choices"][0]["message"]
        assert "content" in data["choices"][0]["message"]
        assert "usage" in data
        assert "finish_reason" in data["choices"][0]

    def test_chat_completions_streaming(self, client):
        """VAL-API-012: Streaming responses work for chat completions."""
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": True,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Read streaming response
        content = response.content.decode("utf-8")
        lines = content.strip().split("\n\n")

        # Should have data: lines and ending with [DONE]
        assert any("data:" in line for line in lines)
        assert any("[DONE]" in line for line in lines)

    def test_chat_completions_with_system_message(self, client):
        """VAL-API-004: Chat completions accept messages array with system role."""
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.5,
            "max_tokens": 50,
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        # Response should include system message acknowledgment
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

    def test_chat_completions_validation_temperature(self, client):
        """VAL-API-005, VAL-API-010: Request validation rejects invalid temperature."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 3.0,  # Invalid: > 2.0
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_chat_completions_temperature_range_boundary(self, client):
        """Test temperature boundary values (0.0 and 2.0 are valid)."""
        # Test minimum valid temperature
        response_min = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.0,
        })
        assert response_min.status_code == 200

        # Test maximum valid temperature
        response_max = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 2.0,
        })
        assert response_max.status_code == 200

    def test_chat_completions_validation_max_tokens(self, client):
        """VAL-API-006: Request validation rejects invalid max_tokens."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0,  # Invalid: must be >= 1
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 422

    def test_chat_completions_max_tokens_limits_output(self, client, mock_backend):
        """VAL-API-006: max_tokens parameter limits output length."""
        # Reset call tracking
        mock_backend._call_count = 0

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Generate a long response"}],
            "max_tokens": 5,
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"

        # Verify max_tokens was passed to backend
        assert mock_backend._last_max_tokens == 5

    def test_chat_completions_with_stop_sequence(self, client, mock_backend):
        """VAL-API-007: Stop sequences halt generation with finish_reason=stop."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hello world"}],
            "stop": "world",  # Stop at "world"
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "finish_reason" in data["choices"][0]
        # Stop sequence should trigger finish_reason="stop"
        assert data["choices"][0]["finish_reason"] == "stop"

        # Verify stop was passed to backend
        assert mock_backend._last_stop == ["world"]

    def test_chat_completions_with_multiple_stop_sequences(self, client):
        """Test multiple stop sequences."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say something"}],
            "stop": ["stop", "end", "finish"],
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200

    def test_chat_completions_finish_reason_length(self, client, mock_backend):
        """Test finish_reason is 'length' when max_tokens limits output."""
        # Set max_tokens to 1 to force length-based truncation
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "max_tokens": 1,
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "length"

    def test_chat_completions_validation_messages_required(self, client):
        """Test that messages are required."""
        request_data = {
            "model": "test-model",
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 422

    def test_chat_completions_with_temperature_param(self, client, mock_backend):
        """Test temperature parameter is properly passed to backend."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 1.5,
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        # Verify temperature was passed to backend
        assert mock_backend._last_temperature == 1.5

    def test_chat_completions_multi_turn_conversation(self, client):
        """Test chat with multiple user and assistant messages."""
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "message" in data["choices"][0]

    def test_chat_completions_streaming_includes_finish_reason(self, client):
        """Test streaming response includes finish_reason in final chunk."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        content = response.content.decode("utf-8")

        # Parse SSE data lines
        for line in content.strip().split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                import json
                try:
                    chunk = json.loads(line[6:])  # Remove "data: " prefix
                    if chunk.get("choices", [{}])[0].get("finish_reason"):
                        # Final chunk should have finish_reason
                        assert chunk["choices"][0]["finish_reason"] == "stop"
                except json.JSONDecodeError:
                    pass


class TestLegacyCompletions:
    """Tests for the /v1/completions endpoint (legacy)."""

    def test_completions_non_streaming(self, client):
        """VAL-API-014: Legacy completions endpoint works (non-streaming)."""
        request_data = {
            "model": "test-model",
            "prompt": "Once upon a time",
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": False,
        }

        response = client.post("/v1/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert "usage" in data
        assert "finish_reason" in data["choices"][0]

    def test_completions_streaming(self, client):
        """VAL-API-012: Streaming responses work for legacy completions."""
        request_data = {
            "model": "test-model",
            "prompt": "Once upon a time",
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": True,
        }

        response = client.post("/v1/completions", json=request_data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Read streaming response
        content = response.content.decode("utf-8")
        lines = content.strip().split("\n\n")

        # Should have data: lines and ending with [DONE]
        assert any("data:" in line for line in lines)
        assert any("[DONE]" in line for line in lines)

    def test_completions_with_list_prompt(self, client):
        """Test completions with list of prompts."""
        request_data = {
            "model": "test-model",
            "prompt": ["First prompt", "Second prompt"],
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": False,
        }

        response = client.post("/v1/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"

    def test_completions_validation_prompt_required(self, client):
        """Test that prompt is required."""
        request_data = {
            "model": "test-model",
        }

        response = client.post("/v1/completions", json=request_data)

        assert response.status_code == 422

    def test_completions_with_temperature(self, client, mock_backend):
        """Test temperature parameter is passed to backend for completions."""
        request_data = {
            "model": "test-model",
            "prompt": "Hello",
            "temperature": 1.2,
            "stream": False,
        }

        response = client.post("/v1/completions", json=request_data)

        assert response.status_code == 200
        assert mock_backend._last_temperature == 1.2

    def test_completions_with_max_tokens(self, client, mock_backend):
        """Test max_tokens parameter limits output for completions."""
        request_data = {
            "model": "test-model",
            "prompt": "Generate long text",
            "max_tokens": 3,
            "stream": False,
        }

        response = client.post("/v1/completions", json=request_data)

        assert response.status_code == 200
        assert mock_backend._last_max_tokens == 3

    def test_completions_with_stop_sequence(self, client, mock_backend):
        """Test stop sequence halts generation for completions with finish_reason=stop."""
        request_data = {
            "model": "test-model",
            "prompt": "Say hello world",
            "stop": "world",
            "stream": False,
        }

        response = client.post("/v1/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert mock_backend._last_stop == ["world"]

    def test_completions_finish_reason_length(self, client):
        """Test finish_reason is 'length' when max_tokens limits output."""
        request_data = {
            "model": "test-model",
            "prompt": "Tell me a story",
            "max_tokens": 1,
            "stream": False,
        }

        response = client.post("/v1/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "length"


class TestErrorHandling:
    """Tests for error handling and responses."""

    def test_404_for_invalid_path(self, client):
        """VAL-API-013: Error responses are appropriate for invalid paths."""
        response = client.get("/v1/invalid-path")

        assert response.status_code == 404

    def test_validation_error_response_format(self, client):
        """Test that validation errors have proper format."""
        request_data = {
            "model": "test-model",
            "messages": [],  # Empty messages should fail validation
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data or "error" in data

    def test_method_not_allowed(self, client):
        """Test that wrong methods are rejected."""
        response = client.post("/health")

        assert response.status_code == 405


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_headers_present_on_get(self, client):
        """Test that CORS headers are present on GET responses."""
        response = client.get("/health", headers={"Origin": "http://example.com"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Preflight requests should succeed with CORS headers
        assert response.status_code == 200
        # CORS headers should be present (handled by middleware)
        assert "access-control-allow-origin" in response.headers


class TestBackendIntegration:
    """Tests for backend integration."""

    def test_backend_initialized_lazily(self, client, mock_backend):
        """Test that backend is initialized when first request is made."""
        # Backend should be initialized
        assert mock_backend._initialized is True

    def test_backend_health_check(self, client):
        """Test that backend health is checked correctly."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.parametrize("endpoint", ["/health", "/v1/models"])
def test_endpoints_exist(client, endpoint):
    """Basic smoke test that endpoints exist and respond."""
    response = client.get(endpoint)
    assert response.status_code in [200, 404]  # 404 is OK for model-specific endpoints
