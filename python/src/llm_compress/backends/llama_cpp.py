"""llama.cpp backend adapter with GGUF conversion support.

This module provides llama.cpp integration with GGUF support for broad hardware
compatibility including CPU inference and quantized models.

References:
    - llama.cpp: https://github.com/ggerganov/llama.cpp
    - GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from typing import Any

from llm_compress.backends.base import BaseBackend

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
    LLAMA_CPP_VERSION = getattr(Llama, "__version__", "unknown")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    LLAMA_CPP_VERSION = None
    Llama = None  # type: ignore

# Try to import transformers for GGUF conversion
try:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    hf_hub_download = None  # type: ignore
    snapshot_download = None  # type: ignore
    list_repo_files = None  # type: ignore


logger = logging.getLogger(__name__)


class GGUFConverter:
    """Converter from HuggingFace models to GGUF format.
    
    This class handles the conversion of HuggingFace models to GGUF format
    using external conversion tools like llama.cpp's convert scripts or
    pre-converted GGUF files from HuggingFace Hub.
    
    Attributes:
        model_id: HuggingFace model identifier
        output_dir: Directory for converted models
        quantization_type: GGUF quantization type (e.g., "Q4_K_M", "Q8_0")
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str | None = None,
        quantization_type: str = "Q4_K_M",
    ) -> None:
        """Initialize the GGUF converter.
        
        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory for converted models (default: temp dir)
            quantization_type: GGUF quantization type
        """
        self.model_id = model_id
        self.output_dir = output_dir or tempfile.gettempdir()
        self.quantization_type = quantization_type

        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)

    def find_preconverted_gguf(self) -> str | None:
        """Search for pre-converted GGUF files on HuggingFace Hub.
        
        Many models have GGUF variants uploaded by the community.
        Common patterns:
        - {model_name}-GGUF
        - {organization}/{model_name}-GGUF
        
        Returns:
            Path to downloaded GGUF file or None if not found
        """
        if not HF_AVAILABLE or list_repo_files is None:
            return None

        # Try common GGUF variant names
        model_name = self.model_id.split("/")[-1]
        gguf_variants = [
            self.model_id.replace("/", "-") + "-GGUF",
            self.model_id + "-GGUF",
            "TheBloke/" + model_name + "-GGUF",
            "TheBloke/" + model_name + "-gguf",
        ]

        for variant in gguf_variants:
            try:
                # List files in the repo
                files = list_repo_files(variant)

                # Look for GGUF files matching our quantization preference
                gguf_files = [f for f in files if f.endswith(".gguf")]

                if gguf_files:
                    # Find best matching quantization
                    preferred_pattern = self.quantization_type.replace("_", r"_")
                    preferred_files = [
                        f for f in gguf_files
                        if re.search(preferred_pattern, f, re.IGNORECASE)
                    ]

                    target_file = preferred_files[0] if preferred_files else gguf_files[0]

                    # Download the file
                    logger.info(f"Found pre-converted GGUF: {variant}/{target_file}")
                    downloaded_path = hf_hub_download(
                        repo_id=variant,
                        filename=target_file,
                        local_dir=self.output_dir,
                        local_dir_use_symlinks=False,
                    )
                    return downloaded_path

            except Exception as e:
                logger.debug(f"Could not check variant {variant}: {e}")
                continue

        return None

    def convert_hf_to_gguf(self) -> str | None:
        """Convert HuggingFace model to GGUF format.
        
        Uses llama.cpp's convert_hf_to_gguf.py script or similar tools.
        Falls back to direct conversion if tools unavailable.
        
        Returns:
            Path to converted GGUF file or None if conversion failed
        """
        if not HF_AVAILABLE:
            raise RuntimeError(
                "Transformers and huggingface-hub required for conversion. "
                "Install with: pip install transformers huggingface-hub"
            )

        # First check for pre-converted versions
        preconverted = self.find_preconverted_gguf()
        if preconverted:
            return preconverted

        # Download the model if needed
        try:
            local_model_path = snapshot_download(
                repo_id=self.model_id,
                local_dir=os.path.join(self.output_dir, "hf_model"),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model {self.model_id}: {e}")

        # Try to use llama.cpp's convert script
        output_gguf = os.path.join(
            self.output_dir,
            f"{self.model_id.split('/')[-1]}_{self.quantization_type}.gguf"
        )

        # Check for convert_hf_to_gguf.py in llama.cpp
        convert_script = self._find_convert_script()

        if convert_script:
            try:
                # Run conversion
                cmd = [
                    "python", convert_script,
                    local_model_path,
                    "--outfile", output_gguf,
                    "--outtype", self._map_quantization_to_outtype(),
                ]

                logger.info(f"Running conversion: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode == 0 and os.path.exists(output_gguf):
                    logger.info(f"Successfully converted to {output_gguf}")
                    return output_gguf
                else:
                    logger.error(f"Conversion failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.error("Conversion timed out")
            except Exception as e:
                logger.error(f"Conversion error: {e}")

        # Fallback: Try using ctransformers or other tools
        logger.warning("Could not convert model using standard tools")
        return None

    def _find_convert_script(self) -> str | None:
        """Find the llama.cpp convert_hf_to_gguf.py script."""
        # Common locations
        possible_paths = [
            "/usr/local/bin/convert_hf_to_gguf.py",
            "/usr/bin/convert_hf_to_gguf.py",
            os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
            os.path.expanduser("~/.local/bin/convert_hf_to_gguf.py"),
        ]

        # Check PATH
        try:
            result = subprocess.run(
                ["which", "convert_hf_to_gguf.py"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                possible_paths.insert(0, result.stdout.strip())
        except Exception:
            pass

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _map_quantization_to_outtype(self, quant_type: str | None = None) -> str:
        """Map GGUF quantization type to llama.cpp outtype."""
        mapping = {
            "Q4_0": "q4_0",
            "Q4_1": "q4_1",
            "Q4_K_S": "q4_k_s",
            "Q4_K_M": "q4_k_m",
            "Q5_0": "q5_0",
            "Q5_1": "q5_1",
            "Q5_K_S": "q5_k_s",
            "Q5_K_M": "q5_k_m",
            "Q6_K": "q6_k",
            "Q8_0": "q8_0",
            "F16": "f16",
            "F32": "f32",
        }
        qt = quant_type or self.quantization_type
        return mapping.get(qt.upper(), "q4_k_m")


class LlamaCppBackend(BaseBackend):
    """llama.cpp inference backend with GGUF support.
    
    This backend uses llama.cpp for broad hardware support,
    including CPU inference and GGUF quantized models.
    
    Features:
    - Native GGUF model loading
    - HuggingFace to GGUF conversion
    - Support for quantized models (Q4, Q5, Q8, etc.)
    - CPU and GPU (CUDA/Metal) inference
    - Streaming and non-streaming generation
    
    Attributes:
        model_id: HuggingFace model identifier or path to GGUF file
        gguf_path: Path to GGUF model file
        llm: llama_cpp.Llama instance (when initialized)
        
    Example:
        >>> backend = LlamaCppBackend(
        ...     model_id="microsoft/DialoGPT-medium",
        ...     quantization="Q4_K_M",
        ... )
        >>> backend.initialize()
        >>> response = backend.generate("Hello, how are you?", max_tokens=50)
    """

    def __init__(
        self,
        model_id: str,
        gguf_path: str | None = None,
        quantization: str = "Q4_K_M",
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize llama.cpp backend.
        
        Args:
            model_id: HuggingFace model identifier or path to GGUF file
            gguf_path: Direct path to GGUF file (skips conversion)
            quantization: GGUF quantization type for conversion
            n_ctx: Context window size
            n_threads: Number of threads (default: auto)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            verbose: Enable verbose output from llama.cpp
            seed: Random seed for reproducibility
            **kwargs: Additional llama.cpp parameters
        """
        super().__init__(model_id, **kwargs)

        self.gguf_path = gguf_path
        self.quantization = quantization
        self.n_ctx = n_ctx
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.seed = seed

        # State
        self.llm: Llama | None = None
        self._initialized = False
        self.converter: GGUFConverter | None = None
        self._is_direct_gguf = model_id.endswith(".gguf") or bool(
            gguf_path and os.path.exists(gguf_path)
        )

    def initialize(self) -> None:
        """Initialize the llama.cpp backend and load the model.
        
        This method:
        1. Checks for llama-cpp-python availability
        2. Converts HuggingFace model to GGUF if needed
        3. Loads the GGUF model with llama.cpp
        
        Raises:
            RuntimeError: If llama-cpp-python is not installed
            RuntimeError: If model loading fails
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )

        try:
            # Determine GGUF path
            if self._is_direct_gguf:
                # Direct GGUF file path
                gguf_file = self.gguf_path or self.model_id
                if not os.path.exists(gguf_file):
                    raise FileNotFoundError(f"GGUF file not found: {gguf_file}")
            else:
                # Need to convert from HuggingFace
                if not HF_AVAILABLE:
                    raise RuntimeError(
                        "HuggingFace libraries required for conversion. "
                        "Install with: pip install transformers huggingface-hub"
                    )

                # Initialize converter
                self.converter = GGUFConverter(
                    model_id=self.model_id,
                    quantization_type=self.quantization,
                )

                # Convert or find pre-converted model
                gguf_file = self.converter.find_preconverted_gguf()

                if not gguf_file:
                    raise RuntimeError(
                        f"Could not find or convert model {self.model_id}. "
                        f"Please provide a direct GGUF file path or install "
                        f"llama.cpp conversion tools."
                    )

            logger.info(f"Loading GGUF model from: {gguf_file}")

            # Load the model with llama.cpp
            self.llm = Llama(
                model_path=gguf_file,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                seed=self.seed,
            )

            self._initialized = True
            logger.info("llama.cpp backend initialized successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize llama.cpp backend: {e}") from e

    def health(self) -> dict[str, Any]:
        """Return backend health status.
        
        Returns:
            Dictionary with status information:
            - status: "healthy" or "unhealthy"
            - backend: "llama.cpp"
            - llama_cpp_available: Whether llama-cpp-python is installed
            - llama_cpp_version: llama-cpp-python version string
            - initialized: Whether the backend is initialized
            - model_id: Model identifier
            - gguf_path: Path to GGUF file
            - quantization: Quantization type used
            - n_ctx: Context window size
            - n_gpu_layers: Number of GPU layers
        """
        health_info: dict[str, Any] = {
            "status": "unhealthy",
            "backend": "llama.cpp",
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "llama_cpp_version": LLAMA_CPP_VERSION,
            "initialized": self._initialized,
            "model_id": self.model_id,
            "gguf_path": self.gguf_path,
            "quantization": self.quantization,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
        }

        if not LLAMA_CPP_AVAILABLE:
            health_info["error"] = "llama-cpp-python not installed"
            return health_info

        if not self._initialized:
            health_info["error"] = "Backend not initialized"
            return health_info

        if self.llm is None:
            health_info["error"] = "LLM instance not created"
            return health_info

        health_info["status"] = "healthy"

        # Add model info if available
        if self.llm:
            try:
                health_info["vocab_size"] = self.llm.n_vocab()
                health_info["context_size"] = self.llm.n_ctx()
            except Exception:
                pass

        return health_info

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Generate text completion.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Whether to stream responses
            
        Returns:
            Generated completion (dict) or stream of chunks (iterator)
            
        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._initialized or self.llm is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        if stream:
            return self._generate_stream(prompt, max_tokens, temperature, top_p, stop)
        else:
            return self._generate_sync(prompt, max_tokens, temperature, top_p, stop)

    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        """Synchronous generation."""
        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        return {
            "id": "llamacpp-gen-" + str(hash(prompt))[:8],
            "object": "text_completion",
            "model": self.model_id,
            "choices": [
                {
                    "text": output["choices"][0]["text"],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": output["choices"][0].get("finish_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": output.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": output.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": output.get("usage", {}).get("total_tokens", 0),
            },
        }

    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> Iterator[dict[str, Any]]:
        """Streaming generation."""
        stream = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=True,
        )

        for i, chunk in enumerate(stream):
            yield {
                "id": f"llamacpp-chunk-{i}",
                "object": "text_completion.chunk",
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk["choices"][0].get("text", ""),
                        "finish_reason": None,
                    }
                ],
            }

        # Final chunk
        yield {
            "id": "llamacpp-chunk-final",
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
        """Generate chat completion.
        
        Args:
            messages: List of chat messages with "role" and "content"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Whether to stream responses
            
        Returns:
            Generated chat completion (dict) or stream (iterator)
            
        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._initialized or self.llm is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # llama.cpp handles chat formatting internally
        if stream:
            return self._chat_stream(messages, max_tokens, temperature, top_p, stop)
        else:
            return self._chat_sync(messages, max_tokens, temperature, top_p, stop)

    def _chat_sync(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        """Synchronous chat generation."""
        output = self.llm.create_chat_completion(
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        return {
            "id": "llamacpp-chat-" + str(hash(str(messages)))[:8],
            "object": "chat.completion",
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output["choices"][0]["message"]["content"],
                    },
                    "finish_reason": output["choices"][0].get("finish_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": output.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": output.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": output.get("usage", {}).get("total_tokens", 0),
            },
        }

    def _chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> Iterator[dict[str, Any]]:
        """Streaming chat generation."""
        stream = self.llm.create_chat_completion(
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=True,
        )

        for i, chunk in enumerate(stream):
            delta = chunk["choices"][0].get("delta", {})
            yield {
                "id": f"llamacpp-chat-chunk-{i}",
                "object": "chat.completion.chunk",
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": delta.get("role", "assistant"),
                            "content": delta.get("content", ""),
                        },
                        "finish_reason": None,
                    }
                ],
            }

        # Final chunk
        yield {
            "id": "llamacpp-chat-chunk-final",
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
        """Shutdown llama.cpp backend and release resources."""
        if self.llm:
            # llama.cpp doesn't have explicit cleanup, but we clear the reference
            self.llm = None

        self._initialized = False
        logger.info("llama.cpp backend shutdown")


class LlamaCppBackendStub(BaseBackend):
    """Stub implementation for when llama-cpp-python is not available."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)

    def initialize(self) -> None:
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Install with: pip install llama-cpp-python"
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "unhealthy",
            "backend": "llama.cpp",
            "llama_cpp_available": False,
            "error": "llama-cpp-python not installed",
        }

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.initialize()
        return None

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        self.initialize()
        return None

    def shutdown(self) -> None:
        pass


# Export the appropriate backend class
if LLAMA_CPP_AVAILABLE:
    LlamaCppBackendClass = LlamaCppBackend
else:
    LlamaCppBackendClass = LlamaCppBackendStub


__all__ = [
    "LlamaCppBackend",
    "LlamaCppBackendClass",
    "LlamaCppBackendStub",
    "GGUFConverter",
    "LLAMA_CPP_AVAILABLE",
    "LLAMA_CPP_VERSION",
]
