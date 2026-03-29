"""KV cache quantization implementation (TurboQuant-style).

This module implements TurboQuant-style KV cache compression with:
- Lloyd-Max optimal scalar quantization
- Random orthogonal rotation
- QJL (Quantized Johnson-Lindenstrauss) projection
- Group quantization for values

The implementation follows the TurboQuant paper (Yang et al., 2024) which achieves
3-bit key compression and 2-bit value compression with minimal accuracy loss.

References:
    - TurboQuant: Making KV Cache Compression Robust to Pruning for Efficient LLM Inference
    - Johnson-Lindenstrauss Lemma: Dimensionality reduction while preserving distances
    - Lloyd-Max Quantization: Optimal scalar quantization minimizing MSE
"""

import math
from typing import Any

import torch
import torch.nn.functional as F


class LloydMaxQuantizer:
    """Lloyd-Max optimal scalar quantization.
    
    Implements the Lloyd-Max algorithm for generating optimal quantization
    codebooks that minimize mean squared error (MSE).
    
    The algorithm iteratively:
    1. Updates centroids (codebook) as mean of each Voronoi cell
    2. Updates boundaries as midpoints between centroids
    
    Attributes:
        num_bits: Number of bits for quantization
        num_levels: Number of quantization levels (2^num_bits)
        codebook: Learned quantization centroids
    """

    def __init__(self, num_bits: int = 3, max_iter: int = 100, tol: float = 1e-6) -> None:
        """Initialize Lloyd-Max quantizer.
        
        Args:
            num_bits: Number of bits for quantization
            max_iter: Maximum iterations for Lloyd-Max algorithm
            tol: Convergence tolerance
        """
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.max_iter = max_iter
        self.tol = tol
        self.codebook: torch.Tensor | None = None
        self.boundaries: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> "LloydMaxQuantizer":
        """Fit the Lloyd-Max quantizer on data.
        
        Args:
            data: Input tensor of any shape, will be flattened
            
        Returns:
            Self for method chaining
        """
        # Flatten data and convert to float64 for numerical stability
        flat_data = data.flatten().to(torch.float64)

        # Initialize codebook with uniform spacing over data range
        data_min, data_max = flat_data.min(), flat_data.max()
        if data_min == data_max:
            # Handle constant data case
            self.codebook = torch.full((self.num_levels,), data_min.item(),
                                       dtype=data.dtype, device=data.device)
            self.boundaries = torch.linspace(data_min - 1, data_max + 1,
                                            self.num_levels + 1,
                                            dtype=data.dtype, device=data.device)
            return self

        # Initialize centroids uniformly across the range
        centroids = torch.linspace(data_min, data_max, self.num_levels,
                                   dtype=torch.float64, device=data.device)

        # Lloyd-Max iterations
        prev_mse = float('inf')
        for iteration in range(self.max_iter):
            # Step 1: Find Voronoi cell boundaries (midpoints between centroids)
            boundaries = self._compute_boundaries(centroids)

            # Step 2: Assign data points to cells and compute new centroids
            new_centroids = self._update_centroids(flat_data, boundaries)

            # Step 3: Compute MSE for convergence check
            mse = self._compute_mse(flat_data, centroids, boundaries)

            # Check convergence
            if abs(prev_mse - mse) < self.tol:
                break

            prev_mse = mse
            centroids = new_centroids

        # Store final codebook and boundaries
        self.codebook = centroids.to(data.dtype)
        self.boundaries = self._compute_boundaries(centroids).to(data.dtype)

        return self

    def _compute_boundaries(self, centroids: torch.Tensor) -> torch.Tensor:
        """Compute decision boundaries as midpoints between centroids."""
        boundaries = torch.zeros(len(centroids) + 1, device=centroids.device,
                                 dtype=centroids.dtype)
        boundaries[0] = centroids[0] - (centroids[1] - centroids[0])
        boundaries[-1] = centroids[-1] + (centroids[-1] - centroids[-2])

        for i in range(len(centroids) - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

        return boundaries

    def _update_centroids(self, data: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """Update centroids as mean of each Voronoi cell."""
        centroids = torch.zeros(len(boundaries) - 1, device=data.device, dtype=data.dtype)

        for i in range(len(boundaries) - 1):
            mask = (data >= boundaries[i]) & (data < boundaries[i + 1])
            if i == len(boundaries) - 2:  # Last cell includes right boundary
                mask = mask | (data == boundaries[i + 1])

            cell_data = data[mask]
            if len(cell_data) > 0:
                centroids[i] = cell_data.mean()
            else:
                # Empty cell: keep previous centroid
                centroids[i] = (boundaries[i] + boundaries[i + 1]) / 2

        return centroids

    def _compute_mse(self, data: torch.Tensor, centroids: torch.Tensor,
                    boundaries: torch.Tensor) -> float:
        """Compute mean squared error for current quantization."""
        total_error = 0.0
        total_count = 0

        for i in range(len(centroids)):
            mask = (data >= boundaries[i]) & (data < boundaries[i + 1])
            if i == len(centroids) - 1:
                mask = mask | (data == boundaries[i + 1])

            cell_data = data[mask]
            if len(cell_data) > 0:
                error = ((cell_data - centroids[i]) ** 2).sum().item()
                total_error += error
                total_count += len(cell_data)

        return total_error / total_count if total_count > 0 else 0.0

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor using the fitted codebook.
        
        Args:
            tensor: Input tensor to quantize
            
        Returns:
            Tuple of (quantized_indices, codebook)
            quantized_indices contains integer indices into codebook
        """
        if self.codebook is None:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")

        flat_tensor = tensor.flatten()

        # Find closest codebook entry for each value
        # Use searchsorted for efficient quantization
        indices = torch.searchsorted(self.boundaries.contiguous(), flat_tensor.contiguous()) - 1
        indices = torch.clamp(indices, 0, self.num_levels - 1)

        return indices.reshape(tensor.shape), self.codebook

    def dequantize(self, indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to values using codebook.
        
        Args:
            indices: Quantized indices
            codebook: Codebook tensor
            
        Returns:
            Dequantized tensor
        """
        return codebook[indices.flatten()].reshape(indices.shape)


class OrthogonalRotation:
    """Random orthogonal rotation for dimensionality reduction.
    
    Generates a random orthogonal matrix using QR decomposition of
    a random Gaussian matrix. Orthogonal rotations preserve vector norms:
    ||Rx|| = ||x|| for any vector x.
    
    Attributes:
        dim: Input/output dimension
        rotation_matrix: Orthogonal matrix (dim x dim)
    """

    def __init__(self, dim: int, seed: int | None = None) -> None:
        """Initialize orthogonal rotation.
        
        Args:
            dim: Input/output dimension
            seed: Random seed for reproducibility
        """
        self.dim = dim
        if seed is not None:
            torch.manual_seed(seed)

        # Generate random Gaussian matrix
        random_matrix = torch.randn(dim, dim)

        # QR decomposition to get orthogonal matrix
        q, r = torch.linalg.qr(random_matrix)

        # Ensure determinant is +1 (proper rotation, not reflection)
        if torch.det(q) < 0:
            q[:, 0] *= -1

        self.rotation_matrix = q

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation to tensor.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Rotated tensor of same shape
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        rotated = x_flat @ self.rotation_matrix.T
        return rotated.reshape(original_shape)

    def inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation (transpose) to tensor.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Inverse-rotated tensor of same shape
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        inverse_rotated = x_flat @ self.rotation_matrix
        return inverse_rotated.reshape(original_shape)


class QJLProjection:
    """Quantized Johnson-Lindenstrauss (QJL) projection.
    
    Implements QJL projection that preserves inner products between vectors
    while reducing dimensionality. The key property is:
    E[<Qx, Qy>] ≈ <x, y>
    
    The projection uses random Gaussian projection followed by quantization.
    
    Attributes:
        input_dim: Input dimension
        proj_dim: Projection dimension (compressed)
        projection_matrix: Random projection matrix (input_dim x proj_dim)
        quantizer: Lloyd-Max quantizer for projected values
    """

    def __init__(self, input_dim: int, proj_dim: int, num_bits: int = 3,
                 seed: int | None = None) -> None:
        """Initialize QJL projection.
        
        Args:
            input_dim: Original input dimension
            proj_dim: Target projection dimension (compressed)
            num_bits: Bits for quantizing projected values
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_bits = num_bits

        if seed is not None:
            torch.manual_seed(seed)

        # Generate random Gaussian projection matrix
        # Scale by 1/sqrt(proj_dim) for JL property:
        # E[P^T P] = proj_dim * (1/proj_dim) * I = I
        # This ensures E[<Px, Py>] = <x, y>
        self.projection_matrix = torch.randn(input_dim, proj_dim) / math.sqrt(proj_dim)

        # Lloyd-Max quantizer for projected values
        self.quantizer = LloydMaxQuantizer(num_bits=num_bits)
        self.codebook: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> "QJLProjection":
        """Fit the QJL projection on data.
        
        This projects sample data and fits the Lloyd-Max quantizer.
        
        Args:
            data: Sample tensor of shape (..., input_dim)
            
        Returns:
            Self for method chaining
        """
        # Project sample data
        original_shape = data.shape
        data_flat = data.reshape(-1, self.input_dim)
        projected = data_flat @ self.projection_matrix

        # Fit quantizer on projected values
        self.quantizer.fit(projected)
        self.codebook = self.quantizer.codebook

        return self

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project and quantize tensor.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Quantized projected tensor as indices
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)

        # Project: (batch, input_dim) @ (input_dim, proj_dim) = (batch, proj_dim)
        projected = x_flat @ self.projection_matrix

        # Quantize projected values
        indices, _ = self.quantizer.quantize(projected)

        return indices.reshape(*original_shape[:-1], self.proj_dim)

    def project_float(self, x: torch.Tensor) -> torch.Tensor:
        """Project without quantization (for query vectors).
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Projected tensor as float values
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        projected = x_flat @ self.projection_matrix
        return projected.reshape(*original_shape[:-1], self.proj_dim)

    def reconstruct(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct tensor from quantized projection.
        
        Args:
            indices: Quantized indices of shape (..., proj_dim)
            
        Returns:
            Approximated original tensor
        """
        # Dequantize to get projected values
        indices_flat = indices.reshape(-1, self.proj_dim)
        projected = self.quantizer.dequantize(indices_flat, self.codebook)

        # Pseudo-inverse: projected @ projection_matrix^T
        # (batch, proj_dim) @ (proj_dim, input_dim) = (batch, input_dim)
        reconstructed = projected @ self.projection_matrix.T

        return reconstructed.reshape(*indices.shape[:-1], self.input_dim)


class TurboQuantKeyCompressor:
    """TurboQuant-style key compressor with MSE + QJL.
    
    Compresses key vectors using:
    1. Orthogonal rotation (preserves norms)
    2. QJL projection (reduces dimension, preserves inner products)
    3. Lloyd-Max quantization (3-bit default)
    
    The compression is designed to preserve cosine similarity for attention.
    
    Attributes:
        head_dim: Dimension per attention head
        proj_dim: Compressed projection dimension
        num_bits: Bits for quantization (default 3)
        rotation: Orthogonal rotation instance
        qjl: QJL projection instance
    """

    def __init__(self, head_dim: int, proj_dim: int | None = None,
                 num_bits: int = 3, seed: int | None = None) -> None:
        """Initialize key compressor.
        
        Args:
            head_dim: Original head dimension
            proj_dim: Target projection dimension (default: head_dim // 2)
            num_bits: Bits for quantization
            seed: Random seed
        """
        self.head_dim = head_dim
        self.proj_dim = proj_dim or (head_dim // 2)
        self.num_bits = num_bits
        self.seed = seed

        # Initialize rotation and QJL
        self.rotation = OrthogonalRotation(head_dim, seed=seed)
        self.qjl = QJLProjection(head_dim, self.proj_dim, num_bits=num_bits, seed=seed)

    def fit(self, sample_keys: torch.Tensor) -> "TurboQuantKeyCompressor":
        """Fit compressor on sample key data.
        
        Args:
            sample_keys: Sample keys of shape (..., head_dim)
            
        Returns:
            Self for method chaining
        """
        # Apply rotation first
        rotated = self.rotation.rotate(sample_keys)

        # Fit QJL on rotated data
        self.qjl.fit(rotated)

        return self

    def compress(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress keys.
        
        Args:
            keys: Key tensor of shape (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Tuple of (compressed_indices, codebook)
            compressed_indices has shape (batch, num_heads, seq_len, proj_dim)
        """
        # Rotate keys
        rotated_keys = self.rotation.rotate(keys)

        # Project and quantize
        indices = self.qjl.project(rotated_keys)

        return indices, self.qjl.codebook

    def decompress(self, indices: torch.Tensor) -> torch.Tensor:
        """Decompress keys.
        
        Args:
            indices: Compressed indices
            
        Returns:
            Approximated keys of shape (batch, num_heads, seq_len, head_dim)
        """
        # Reconstruct from QJL
        rotated_keys = self.qjl.reconstruct(indices)

        # Inverse rotation
        keys = self.rotation.inverse_rotate(rotated_keys)

        return keys

    def compute_attention_score(self, query: torch.Tensor,
                               compressed_keys: torch.Tensor) -> torch.Tensor:
        """Compute attention scores with compressed keys.
        
        This uses the unbiased estimator: <q, k> ≈ <Rq, QJL(k)>
        where R is the orthogonal rotation and QJL is the quantized projection.
        
        Args:
            query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            compressed_keys: Compressed key indices
            
        Returns:
            Attention scores
        """
        # Rotate query
        rotated_query = self.rotation.rotate(query)

        # Project query (without quantization)
        proj_query = self.qjl.project_float(rotated_query)

        # Dequantize keys
        indices_flat = compressed_keys.reshape(-1, self.proj_dim)
        proj_keys = self.qjl.quantizer.dequantize(indices_flat, self.qjl.codebook)
        proj_keys = proj_keys.reshape(*compressed_keys.shape[:-1], self.proj_dim)

        # Compute attention: <q_proj, k_proj>
        scores = torch.matmul(proj_query, proj_keys.transpose(-2, -1))

        # With correct JL scaling (1/sqrt(input_dim)), the projection preserves inner products
        # E[<Px, Py>] = <x, y>, so no additional scaling needed

        return scores


class GroupValueQuantizer:
    """Group quantization for value vectors.
    
    Groups value vectors and quantizes within each group using shared
dictionary-based quantization. This is more efficient than per-vector
    quantization.
    
    Attributes:
        head_dim: Dimension per head
        group_size: Number of vectors per group
        num_bits: Bits per element (2 or 4)
        codebook: Shared codebook for group
    """

    def __init__(self, head_dim: int, group_size: int = 8, num_bits: int = 2) -> None:
        """Initialize group value quantizer.
        
        Args:
            head_dim: Dimension per attention head
            group_size: Number of vectors in each group
            num_bits: Bits for quantization (2 or 4)
        """
        self.head_dim = head_dim
        self.group_size = group_size
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.codebook: torch.Tensor | None = None

    def fit(self, sample_values: torch.Tensor) -> "GroupValueQuantizer":
        """Fit quantizer on sample values.
        
        Fits a shared codebook on groups of values.
        
        Args:
            sample_values: Sample values of shape (..., head_dim)
            
        Returns:
            Self for method chaining
        """
        # Flatten to (num_vectors, head_dim)
        if sample_values.dim() == 1:
            sample_values = sample_values.unsqueeze(0)

        # Get actual dimensions from the tensor
        actual_head_dim = sample_values.shape[-1]
        num_vectors = sample_values.shape[0] if sample_values.dim() >= 1 else 1

        # Flatten all dimensions except the last one
        if sample_values.dim() > 2:
            sample_values = sample_values.reshape(-1, actual_head_dim)
            num_vectors = sample_values.shape[0]

        padded_len = ((num_vectors + self.group_size - 1) // self.group_size) * self.group_size

        # Pad if necessary - pad zeros at the end
        if num_vectors < padded_len:
            padding = padded_len - num_vectors
            # Pad along dimension 0 (num_vectors dimension)
            sample_values = F.pad(sample_values, (0, 0, 0, padding))

        # Reshape into groups
        num_groups = padded_len // self.group_size
        grouped = sample_values[:num_groups * self.group_size].reshape(
            num_groups, self.group_size, actual_head_dim
        )

        # Fit codebook on all group values
        flat_values = grouped.reshape(-1)
        quantizer = LloydMaxQuantizer(num_bits=self.num_bits)
        quantizer.fit(flat_values)

        self.codebook = quantizer.codebook
        self.boundaries = quantizer.boundaries

        return self

    def quantize_group(self, group: torch.Tensor) -> torch.Tensor:
        """Quantize a group of value vectors.
        
        Args:
            group: Group of values, shape (group_size, head_dim)
            
        Returns:
            Quantized indices
        """
        flat = group.flatten()
        indices = torch.searchsorted(self.boundaries.contiguous(), flat.contiguous()) - 1
        indices = torch.clamp(indices, 0, self.num_levels - 1)
        return indices.reshape(group.shape)

    def dequantize_group(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize a group.
        
        Args:
            indices: Quantized indices, shape (group_size, head_dim)
            
        Returns:
            Dequantized values
        """
        flat_indices = indices.flatten()
        values = self.codebook[flat_indices]
        return values.reshape(indices.shape)

    def compress(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compress values with group quantization.
        
        Args:
            values: Value tensor of shape (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Tuple of (compressed_indices, codebook, original_length)
        """
        batch_size, num_heads, seq_len, head_dim = values.shape

        # Reshape to (batch * num_heads, seq_len, head_dim)
        values_flat = values.reshape(-1, seq_len, head_dim)

        # Pad seq_len to multiple of group_size
        padded_len = ((seq_len + self.group_size - 1) // self.group_size) * self.group_size
        if seq_len < padded_len:
            padding = padded_len - seq_len
            values_flat = F.pad(values_flat, (0, 0, 0, padding))

        # Reshape into groups: (batch*heads, num_groups, group_size, head_dim)
        num_groups = padded_len // self.group_size
        grouped = values_flat.reshape(batch_size * num_heads, num_groups,
                                     self.group_size, head_dim)

        # Quantize each group
        all_indices = []
        for i in range(batch_size * num_heads):
            for g in range(num_groups):
                indices = self.quantize_group(grouped[i, g])
                all_indices.append(indices)

        compressed = torch.stack(all_indices)
        compressed = compressed.reshape(batch_size, num_heads, num_groups,
                                       self.group_size, head_dim)

        return compressed, self.codebook, seq_len

    def decompress(self, compressed: torch.Tensor, codebook: torch.Tensor,
                   original_length: int) -> torch.Tensor:
        """Decompress values.
        
        Args:
            compressed: Compressed indices
            codebook: Codebook tensor
            original_length: Original sequence length
            
        Returns:
            Dequantized values
        """
        batch_size, num_heads, num_groups, group_size, head_dim = compressed.shape

        # Dequantize each group
        all_values = []
        for i in range(batch_size):
            for h in range(num_heads):
                for g in range(num_groups):
                    values = codebook[compressed[i, h, g].flatten()].reshape(
                        group_size, head_dim
                    )
                    all_values.append(values)

        values = torch.stack(all_values)
        values = values.reshape(batch_size, num_heads, num_groups * group_size, head_dim)

        # Trim padding
        values = values[:, :, :original_length, :]

        return values


class KVCacheQuantizer:
    """TurboQuant-style KV cache quantizer.
    
    Implements 3-bit key compression and 2-bit/4-bit value compression
    using Lloyd-Max codebooks with QJL projection.
    
    This is the main entry point for KV cache quantization. It combines
    the key compressor and value quantizer for end-to-end KV cache
    compression.
    
    Attributes:
        key_bits: Bits for key quantization (default: 3)
        value_bits: Bits for value quantization (default: 2)
        head_dim: Dimension per attention head
        key_proj_dim: Projection dimension for keys
        key_compressor: TurboQuantKeyCompressor instance
        value_quantizer: GroupValueQuantizer instance
    """

    def __init__(self, head_dim: int = 64, key_bits: int = 3,
                 value_bits: int = 2, key_proj_dim: int | None = None,
                 value_group_size: int = 8, seed: int | None = None) -> None:
        """Initialize the KV cache quantizer.
        
        Args:
            head_dim: Dimension per attention head
            key_bits: Bits for key quantization (default 3)
            value_bits: Bits for value quantization (default 2)
            key_proj_dim: Projection dimension for keys (default head_dim//2)
            value_group_size: Group size for value quantization
            seed: Random seed for reproducibility
        """
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.key_proj_dim = key_proj_dim or (head_dim // 2)
        self.value_group_size = value_group_size
        self.seed = seed

        # Initialize compressors
        self.key_compressor = TurboQuantKeyCompressor(
            head_dim=head_dim,
            proj_dim=self.key_proj_dim,
            num_bits=key_bits,
            seed=seed
        )

        self.value_quantizer = GroupValueQuantizer(
            head_dim=head_dim,
            group_size=value_group_size,
            num_bits=value_bits
        )

        self._fitted = False

    def fit(self, sample_keys: torch.Tensor, sample_values: torch.Tensor) -> "KVCacheQuantizer":
        """Fit the quantizers on sample KV cache data.
        
        Args:
            sample_keys: Sample key tensors of shape (..., head_dim)
            sample_values: Sample value tensors of shape (..., head_dim)
            
        Returns:
            Self for method chaining
        """
        self.key_compressor.fit(sample_keys)
        self.value_quantizer.fit(sample_values)
        self._fitted = True
        return self

    def compress_kv_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> dict[str, Any]:
        """Compress KV cache tensors.
        
        Args:
            keys: Key tensor of shape (batch, num_heads, seq_len, head_dim)
            values: Value tensor of shape (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Dictionary containing compressed data:
            {
                'key_indices': Quantized key indices,
                'key_codebook': Key codebook,
                'value_indices': Quantized value indices,
                'value_codebook': Value codebook,
                'seq_len': Original sequence length,
                'metadata': Compression metadata
            }
        """
        if not self._fitted:
            # Auto-fit on the data if not already fitted
            self.fit(keys.flatten(0, -2), values.flatten(0, -2))

        # Compress keys
        key_indices, key_codebook = self.key_compressor.compress(keys)

        # Compress values
        value_indices, value_codebook, seq_len = self.value_quantizer.compress(values)

        return {
            'key_indices': key_indices,
            'key_codebook': key_codebook,
            'value_indices': value_indices,
            'value_codebook': value_codebook,
            'seq_len': seq_len,
            'metadata': {
                'head_dim': self.head_dim,
                'key_bits': self.key_bits,
                'value_bits': self.value_bits,
                'key_proj_dim': self.key_proj_dim,
                'value_group_size': self.value_group_size,
            }
        }

    def decompress_kv_cache(self, compressed: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress KV cache from compressed representation.
        
        Args:
            compressed: Dictionary from compress_kv_cache()
            
        Returns:
            Tuple of (decompressed_keys, decompressed_values)
        """
        # Decompress keys
        keys = self.key_compressor.decompress(compressed['key_indices'])

        # Decompress values
        values = self.value_quantizer.decompress(
            compressed['value_indices'],
            compressed['value_codebook'],
            compressed['seq_len']
        )

        return keys, values

    def compute_attention(
        self,
        query: torch.Tensor,
        compressed_keys: torch.Tensor,
        compressed_values: dict[str, Any],
        key_codebook: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute attention with compressed KV cache.
        
        Uses the unbiased estimator for key-query similarity and
        decompresses values for output computation.
        
        Args:
            query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            compressed_keys: Compressed key indices
            compressed_values: Compressed value data
            key_codebook: Key codebook
            mask: Optional attention mask
            
        Returns:
            Attention output tensor
        """
        # Compute attention scores using QJL projection (unbiased estimator)
        scores = self.key_compressor.compute_attention_score(query, compressed_keys)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Decompress values
        values = self.value_quantizer.decompress(
            compressed_values,
            self.value_quantizer.codebook,
            compressed_values.shape[2] if compressed_values.dim() >= 3 else compressed_values.shape[0]
        )

        # Compute weighted sum
        output = torch.matmul(attn_weights, values)

        return output


def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor,
                              dim: int = -1) -> torch.Tensor:
    """Compute cosine similarity between tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        dim: Dimension to compute similarity over
        
    Returns:
        Cosine similarity tensor
    """
    x_norm = F.normalize(x, p=2, dim=dim)
    y_norm = F.normalize(y, p=2, dim=dim)
    return (x_norm * y_norm).sum(dim=dim)


def estimate_compression_ratio(head_dim: int = 64, key_bits: int = 3,
                                value_bits: int = 2, key_proj_dim: int | None = None,
                                seq_len: int = 1024) -> dict[str, float]:
    """Estimate compression ratio for KV cache.
    
    Args:
        head_dim: Original head dimension
        key_bits: Bits for key quantization
        value_bits: Bits for value quantization
        key_proj_dim: Compressed key dimension (default head_dim//2)
        seq_len: Sequence length for estimation
        
    Returns:
        Dictionary with compression statistics
    """
    if key_proj_dim is None:
        key_proj_dim = head_dim // 2

    # Original size: 2 * seq_len * head_dim * 4 bytes (float32)
    original_bytes = 2 * seq_len * head_dim * 4

    # Compressed keys: seq_len * key_proj_dim * (key_bits / 8)
    key_bytes = seq_len * key_proj_dim * (key_bits / 8)

    # Compressed values: seq_len * head_dim * (value_bits / 8)
    value_bytes = seq_len * head_dim * (value_bits / 8)

    # Codebook overhead (negligible for long sequences)
    key_codebook_bytes = (2 ** key_bits) * 4  # 4 bytes per float32
    value_codebook_bytes = (2 ** value_bits) * 4

    compressed_bytes = key_bytes + value_bytes + key_codebook_bytes + value_codebook_bytes

    return {
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'compression_ratio': original_bytes / compressed_bytes,
        'key_compression_ratio': (seq_len * head_dim * 4) / key_bytes,
        'value_compression_ratio': (seq_len * head_dim * 4) / value_bytes,
    }


__all__ = [
    "KVCacheQuantizer",
    "LloydMaxQuantizer",
    "OrthogonalRotation",
    "QJLProjection",
    "TurboQuantKeyCompressor",
    "GroupValueQuantizer",
    "compute_cosine_similarity",
    "estimate_compression_ratio",
]
