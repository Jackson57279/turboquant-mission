"""Weight quantization implementation.

This module implements block-wise weight quantization using bitsandbytes,
supporting 4-bit and 8-bit quantization schemes.

Quantization schemes:
    - 4-bit: NF4 (Normal Float 4) with block-wise quantization
    - 8-bit: FP8/INT8 with block-wise quantization

Custom format:
    Models are saved in a custom format using safetensors for efficient
    storage and loading. Metadata includes quantization parameters.

References:
    - AirLLM: Compress your LLM using layer-wise short GPTQ and layer-wise streaming
    - Dettmers et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers"
    - Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs"
"""

import json
from pathlib import Path
from typing import Any

import bitsandbytes as bnb
import torch
from safetensors.torch import load_file, save_file

from llm_compress.download import get_model_dir, load_metadata, save_metadata


def quantize_tensor(tensor: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, Any, tuple]:
    """Quantize a single tensor to specified bit width.
    
    Args:
        tensor: Input tensor to quantize (must be 2D or can be reshaped to 2D)
        bits: Quantization bit width (4 or 8)
        
    Returns:
        Tuple of (quantized_tensor, quantization_state, original_shape)
        
    Raises:
        ValueError: If bits is not 4 or 8
        
    Example:
        >>> weight = torch.randn(512, 512)
        >>> qweight, qstate, shape = quantize_tensor(weight, bits=4)
        >>> print(qweight.dtype)  # torch.uint8 (packed 4-bit values)
    """
    if bits not in [4, 8]:
        raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")

    original_shape = tensor.shape

    # Ensure tensor is 2D and contiguous
    if tensor.dim() == 1:
        # Reshape 1D to 2D with single row
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() > 2:
        # Flatten dimensions beyond the last two
        tensor = tensor.reshape(-1, tensor.shape[-1])

    tensor = tensor.contiguous().float()

    if bits == 4:
        # Use NF4 (Normal Float 4) quantization
        quant_type = bnb.functional.get_4bit_type('nf4', device='cpu')
        qweight, qstate = bnb.functional.quantize_4bit(tensor, quant_type)
    else:
        # Use 8-bit block-wise quantization
        qweight, qstate = bnb.functional.quantize_blockwise(tensor)

    return qweight, qstate, original_shape


def dequantize_tensor(
    quantized: torch.Tensor,
    qstate: Any,
    original_shape: tuple,
    bits: int = 4
) -> torch.Tensor:
    """Dequantize a tensor from its quantized representation.
    
    Args:
        quantized: Quantized tensor data
        qstate: Quantization state (absmax, blocksize, etc.)
        original_shape: Original shape of the tensor before quantization
        bits: Quantization bit width (4 or 8)
        
    Returns:
        Dequantized tensor with original shape
        
    Raises:
        ValueError: If bits is not 4 or 8
    """
    if bits not in [4, 8]:
        raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")

    if bits == 4:
        dequantized = bnb.functional.dequantize_4bit(quantized, qstate, original_shape)
    else:
        # For 8-bit, we need to reshape back after dequantization
        dequantized = bnb.functional.dequantize_blockwise(quantized, qstate)

    return dequantized.reshape(original_shape)


def quantize_model_state_dict(
    state_dict: dict[str, torch.Tensor],
    bits: int = 4,
    quantize_linear_only: bool = True
) -> dict[str, Any]:
    """Quantize all weight tensors in a model state dict.
    
    Args:
        state_dict: Model state dictionary from HuggingFace
        bits: Quantization bit width (4 or 8)
        quantize_linear_only: If True, only quantize linear layer weights
            (those containing 'weight' in name and are 2D+ tensors)
            
    Returns:
        Dictionary containing:
            - quantized_tensors: dict of quantized weight tensors
            - quantization_metadata: dict with per-tensor quantization info
            - non_quantized: dict of tensors that weren't quantized
            
    Note:
        Non-weight tensors (embeddings, biases, layer norms, etc.) are not
        quantized to preserve model accuracy.
    """
    quantized_tensors = {}
    quantization_metadata = {}
    non_quantized = {}

    for name, tensor in state_dict.items():
        # Decide whether to quantize this tensor
        should_quantize = (
            tensor.dtype in [torch.float32, torch.float16, torch.bfloat16] and
            tensor.numel() > 0 and
            (not quantize_linear_only or
             ('weight' in name.lower() and tensor.dim() >= 2))
        )

        if should_quantize:
            try:
                qweight, qstate, orig_shape = quantize_tensor(tensor, bits)
                quantized_tensors[name] = qweight
                quantization_metadata[name] = {
                    'bits': bits,
                    'shape': orig_shape,
                    'dtype': str(tensor.dtype).replace('torch.', ''),
                    'absmax': qstate.absmax.tolist() if hasattr(qstate, 'absmax') else None,
                    'blocksize': qstate.blocksize if hasattr(qstate, 'blocksize') else 64,
                    'quantized': True,
                    'code': qstate.code if hasattr(qstate, 'code') else None,
                }
            except Exception as e:
                # If quantization fails, keep original
                non_quantized[name] = tensor
                quantization_metadata[name] = {'quantized': False, 'error': str(e)}
        else:
            non_quantized[name] = tensor
            quantization_metadata[name] = {'quantized': False, 'reason': 'excluded'}

    return {
        'quantized_tensors': quantized_tensors,
        'quantization_metadata': quantization_metadata,
        'non_quantized': non_quantized,
    }


def save_quantized_model(
    quantized_data: dict[str, Any],
    output_dir: str | Path,
    model_id: str,
    bits: int = 4
) -> Path:
    """Save quantized model to custom format using safetensors.
    
    The custom format consists of:
        - model.safetensors: Quantized and non-quantized tensors
        - quantization_config.json: Quantization metadata and config
        
    Args:
        quantized_data: Output from quantize_model_state_dict()
        output_dir: Directory to save the quantized model
        model_id: Original model identifier
        bits: Quantization bit width
        
    Returns:
        Path to the output directory
        
    Note:
        Safetensors format provides fast loading and is safe from pickle
        deserialization attacks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all tensors into one dict for safetensors
    all_tensors = {}
    all_tensors.update(quantized_data['quantized_tensors'])
    all_tensors.update(quantized_data['non_quantized'])

    # Save tensors using safetensors
    model_path = output_dir / 'model.safetensors'
    save_file(all_tensors, str(model_path))

    # Save quantization metadata and config
    config = {
        'model_id': model_id,
        'bits': bits,
        'quantization_type': 'nf4' if bits == 4 else 'int8',
        'quantization_metadata': quantized_data['quantization_metadata'],
        'format_version': '1.0',
    }

    config_path = output_dir / 'quantization_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return output_dir


def load_quantized_model(model_dir: str | Path) -> dict[str, Any]:
    """Load a quantized model from custom format.
    
    Args:
        model_dir: Directory containing the quantized model
        
    Returns:
        Dictionary with:
            - tensors: dict of loaded tensors (quantized and non-quantized)
            - config: quantization configuration
            - quantized_names: list of quantized tensor names
            
    Raises:
        FileNotFoundError: If model files don't exist
        ValueError: If format version is incompatible
    """
    model_dir = Path(model_dir)

    # Load quantization config
    config_path = model_dir / 'quantization_config.json'
    with open(config_path) as f:
        config = json.load(f)

    if config.get('format_version') != '1.0':
        raise ValueError(f"Unsupported format version: {config.get('format_version')}")

    # Load tensors using safetensors
    model_path = model_dir / 'model.safetensors'
    tensors = load_file(str(model_path))

    # Identify quantized tensors
    quantized_names = [
        name for name, meta in config['quantization_metadata'].items()
        if meta.get('quantized', False)
    ]

    return {
        'tensors': tensors,
        'config': config,
        'quantized_names': quantized_names,
    }


def dequantize_model(
    model_data: dict[str, Any],
    dequantize_to_dtype: torch.dtype | None = None
) -> dict[str, torch.Tensor]:
    """Dequantize all quantized tensors in a loaded model.
    
    Args:
        model_data: Output from load_quantized_model()
        dequantize_to_dtype: Target dtype for dequantized tensors (default: float32)
        
    Returns:
        State dict with all tensors dequantized to specified dtype
    """
    if dequantize_to_dtype is None:
        dequantize_to_dtype = torch.float32

    config = model_data['config']
    metadata = config['quantization_metadata']
    tensors = model_data['tensors']

    dequantized = {}

    for name, tensor in tensors.items():
        meta = metadata.get(name, {})

        if meta.get('quantized', False):
            # Reconstruct quantization state
            bits = meta['bits']
            shape = tuple(meta['shape'])

            # Build QuantState object
            absmax = torch.tensor(meta['absmax'], dtype=torch.float32)
            qstate = bnb.functional.QuantState(
                absmax=absmax,
                blocksize=meta.get('blocksize', 64),
                code=meta.get('code'),
            )

            # Dequantize
            dequantized[name] = dequantize_tensor(
                tensor, qstate, shape, bits
            ).to(dequantize_to_dtype)
        else:
            # Non-quantized tensor - just convert dtype
            dequantized[name] = tensor.to(dequantize_to_dtype)

    return dequantized


def quantize_model(
    model_id: str,
    bits: int = 4,
    cache_dir: str | None = None,
    output_suffix: str | None = None
) -> Path:
    """Quantize a downloaded HuggingFace model.
    
    This is the main entry point for model quantization from the CLI.
    
    Args:
        model_id: HuggingFace model identifier (e.g., 'microsoft/DialoGPT-medium')
        bits: Quantization bit width (4 or 8)
        cache_dir: Custom cache directory (default: ~/.cache/llm-compress)
        output_suffix: Suffix for output directory (default: 'quantized-{bits}bit')
        
    Returns:
        Path to the quantized model directory
        
    Raises:
        ValueError: If model not found in cache or bits not supported
        RuntimeError: If quantization fails
        
    Example:
        >>> from llm_compress.quantization import quantize_model
        >>> quantized_path = quantize_model('microsoft/DialoGPT-medium', bits=4)
        >>> print(f"Quantized model saved to: {quantized_path}")
    """
    if bits not in [4, 8]:
        raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")

    # Get model directory
    model_dir = get_model_dir(model_id, cache_dir)
    original_dir = model_dir / 'original'

    if not original_dir.exists():
        raise ValueError(
            f"Model {model_id} not found in cache. "
            "Please download it first with: llm-compress download {model_id}"
        )

    # Determine output directory
    if output_suffix is None:
        output_suffix = f'quantized-{bits}bit'
    output_dir = model_dir / output_suffix

    # Load metadata
    metadata = load_metadata(model_id, cache_dir)

    # Load original model state dict
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(original_dir),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    state_dict = model.state_dict()

    # Quantize
    quantized_data = quantize_model_state_dict(state_dict, bits)

    # Save quantized model
    save_quantized_model(quantized_data, output_dir, model_id, bits)

    # Update metadata
    metadata['quantized'] = True
    metadata['quantization'] = {
        'bits': bits,
        'type': 'nf4' if bits == 4 else 'int8',
        'quantized_tensors': len(quantized_data['quantized_tensors']),
        'non_quantized_tensors': len(quantized_data['non_quantized']),
    }
    save_metadata(model_id, metadata, cache_dir)

    # Clean up loaded model to free memory
    del model
    import gc
    gc.collect()

    return output_dir


def get_compression_ratio(model_dir: str | Path) -> float:
    """Calculate the compression ratio of a quantized model.
    
    Args:
        model_dir: Directory containing the quantized model
        
    Returns:
        Compression ratio (original_size / quantized_size)
        
    Note:
        This estimates the original size based on the quantized metadata.
        For accurate measurement, compare against the original model files.
    """
    model_data = load_quantized_model(model_dir)
    config = model_data['config']
    metadata = config['quantization_metadata']

    total_original_bytes = 0
    total_quantized_bytes = 0

    for name, tensor in model_data['tensors'].items():
        meta = metadata.get(name, {})
        tensor_bytes = tensor.numel() * tensor.element_size()

        if meta.get('quantized', False):
            # Original was float32 (4 bytes per element)
            original_bytes = tensor.numel() * 4
            if meta['bits'] == 4:
                # 4-bit quantization packs 2 values per byte
                # but has overhead from absmax values
                quantized_bytes = tensor_bytes
            else:
                # 8-bit quantization: 1 byte per element + absmax overhead
                quantized_bytes = tensor_bytes
        else:
            original_bytes = tensor_bytes
            quantized_bytes = tensor_bytes

        total_original_bytes += original_bytes
        total_quantized_bytes += quantized_bytes

    return total_original_bytes / total_quantized_bytes


def estimate_accuracy_loss(bits: int) -> float:
    """Estimate expected accuracy loss from quantization.
    
    Args:
        bits: Quantization bit width (4 or 8)
        
    Returns:
        Estimated accuracy loss percentage
        
    Note:
        These are approximate values based on literature:
        - 4-bit: typically <1% accuracy loss
        - 8-bit: typically <0.5% accuracy loss
    """
    if bits == 4:
        return 0.5  # 0.5% estimated loss (conservative, usually <1%)
    elif bits == 8:
        return 0.2  # 0.2% estimated loss (conservative, usually <0.5%)
    else:
        return 1.0  # Unknown, conservative estimate
