/**
 * KV cache quantization implementation (TurboQuant-style).
 *
 * This module implements KV cache compression using:
 * - Lloyd-Max optimal scalar quantization
 * - Random orthogonal rotation
 * - QJL (Quantized Johnson-Lindenstrauss) projection
 * - 3-bit keys + 2-bit values compression
 *
 * @module quantization/kv_cache
 */

/**
 * Compute cosine similarity between two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Cosine similarity (0-1)
 */
export function computeCosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Estimate the compression ratio for KV cache.
 *
 * @param keyBits - Bit width for key compression (default: 3)
 * @param valueBits - Bit width for value compression (default: 2)
 * @param originalBits - Original bit width (default: 16 for FP16)
 * @returns Compression ratio
 */
export function estimateCompressionRatio(
  keyBits = 3,
  valueBits = 2,
  originalBits = 16
): number {
  const totalCompressedBits = keyBits + valueBits;
  return originalBits / totalCompressedBits;
}

/**
 * Lloyd-Max quantizer for optimal quantization levels.
 *
 * This implements Lloyd's algorithm to find optimal quantization
 * levels that minimize mean squared error.
 */
export class LloydMaxQuantizer {
  private levels: number;
  private iterations: number;

  /**
   * Create a Lloyd-Max quantizer.
   *
   * @param levels - Number of quantization levels
   * @param iterations - Number of Lloyd iterations (default: 20)
   */
  constructor(levels: number, iterations = 20) {
    this.levels = levels;
    this.iterations = iterations;
  }

  /**
   * Generate optimal quantization codebook.
   *
   * @param data - Training data to optimize for
   * @returns Quantization levels (codebook)
   */
  generateCodebook(data: Float32Array): Float32Array {
    // Initialize with uniform spacing
    const min = Math.min(...data);
    const max = Math.max(...data);
    const step = (max - min) / (this.levels - 1);

    const codebook = new Float32Array(this.levels);
    for (let i = 0; i < this.levels; i++) {
      codebook[i] = min + i * step;
    }

    // Lloyd iterations
    for (let iter = 0; iter < this.iterations; iter++) {
      // Assign each data point to nearest centroid
      const assignments: number[][] = Array.from({ length: this.levels }, () => []);

      for (let i = 0; i < data.length; i++) {
        let bestIdx = 0;
        let bestDist = Math.abs(data[i] - codebook[0]);

        for (let j = 1; j < this.levels; j++) {
          const dist = Math.abs(data[i] - codebook[j]);
          if (dist < bestDist) {
            bestDist = dist;
            bestIdx = j;
          }
        }

        assignments[bestIdx].push(i);
      }

      // Update centroids
      for (let i = 0; i < this.levels; i++) {
        if (assignments[i].length > 0) {
          let sum = 0;
          for (const idx of assignments[i]) {
            sum += data[idx];
          }
          codebook[i] = sum / assignments[i].length;
        }
      }
    }

    return codebook;
  }

  /**
   * Quantize data using the codebook.
   *
   * @param data - Input data
   * @param codebook - Quantization levels
   * @returns Quantized indices
   */
  quantize(data: Float32Array, codebook: Float32Array): Uint8Array {
    const indices = new Uint8Array(data.length);

    for (let i = 0; i < data.length; i++) {
      let bestIdx = 0;
      let bestDist = Math.abs(data[i] - codebook[0]);

      for (let j = 1; j < codebook.length; j++) {
        const dist = Math.abs(data[i] - codebook[j]);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = j;
        }
      }

      indices[i] = bestIdx;
    }

    return indices;
  }

  /**
   * Dequantize indices using the codebook.
   *
   * @param indices - Quantized indices
   * @param codebook - Quantization levels
   * @returns Dequantized data
   */
  dequantize(indices: Uint8Array, codebook: Float32Array): Float32Array {
    const data = new Float32Array(indices.length);

    for (let i = 0; i < indices.length; i++) {
      data[i] = codebook[indices[i]];
    }

    return data;
  }
}

/**
 * Orthogonal rotation matrix generator.
 *
 * Creates random orthogonal matrices using the Haar measure.
 */
export class OrthogonalRotation {
  private dimension: number;

  /**
   * Create an orthogonal rotation generator.
   *
   * @param dimension - Vector dimension
   */
  constructor(dimension: number) {
    this.dimension = dimension;
  }

  /**
   * Generate a random orthogonal matrix using QR decomposition.
   *
   * @returns Orthogonal rotation matrix
   */
  generate(): Float32Array {
    // Generate random matrix using Box-Muller transform for normal distribution
    const matrix = new Float32Array(this.dimension * this.dimension);

    for (let i = 0; i < this.dimension * this.dimension; i++) {
      // Box-Muller transform
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      matrix[i] = z;
    }

    // Gram-Schmidt orthogonalization
    const orthogonal: Float32Array[] = [];

    for (let i = 0; i < this.dimension; i++) {
      let vec = matrix.slice(i * this.dimension, (i + 1) * this.dimension);

      // Subtract projections onto previous vectors
      for (const prevVec of orthogonal) {
        let projection = 0;
        for (let j = 0; j < this.dimension; j++) {
          projection += vec[j] * prevVec[j];
        }
        for (let j = 0; j < this.dimension; j++) {
          vec[j] -= projection * prevVec[j];
        }
      }

      // Normalize
      let norm = 0;
      for (let j = 0; j < this.dimension; j++) {
        norm += vec[j] * vec[j];
      }
      norm = Math.sqrt(norm);

      for (let j = 0; j < this.dimension; j++) {
        vec[j] /= norm;
      }

      orthogonal.push(vec);
    }

    // Flatten back to single array
    const result = new Float32Array(this.dimension * this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      result.set(orthogonal[i], i * this.dimension);
    }

    return result;
  }

  /**
   * Apply rotation to a vector.
   *
   * @param matrix - Orthogonal rotation matrix
   * @param vector - Input vector
   * @returns Rotated vector
   */
  apply(matrix: Float32Array, vector: Float32Array): Float32Array {
    const result = new Float32Array(this.dimension);

    for (let i = 0; i < this.dimension; i++) {
      let sum = 0;
      for (let j = 0; j < this.dimension; j++) {
        sum += matrix[i * this.dimension + j] * vector[j];
      }
      result[i] = sum;
    }

    return result;
  }

  /**
   * Verify that a matrix is orthogonal (preserves norms).
   *
   * @param matrix - Matrix to verify
   * @returns True if matrix is orthogonal
   */
  verifyOrthogonal(matrix: Float32Array): boolean {
    const tolerance = 1e-6;

    // Check R * R^T = I
    for (let i = 0; i < this.dimension; i++) {
      for (let j = 0; j < this.dimension; j++) {
        let dot = 0;
        for (let k = 0; k < this.dimension; k++) {
          dot += matrix[i * this.dimension + k] * matrix[j * this.dimension + k];
        }

        const expected = i === j ? 1 : 0;
        if (Math.abs(dot - expected) > tolerance) {
          return false;
        }
      }
    }

    return true;
  }
}

/**
 * QJL (Quantized Johnson-Lindenstrauss) projection.
 *
 * Projects high-dimensional vectors to lower dimensions while
 * approximately preserving inner products.
 */
export class QJLProjection {
  private inputDim: number;
  private outputDim: number;
  private projectionMatrix: Float32Array | null = null;

  /**
   * Create a QJL projection.
   *
   * @param inputDim - Input dimension
   * @param outputDim - Output dimension (projected)
   */
  constructor(inputDim: number, outputDim: number) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
  }

  /**
   * Initialize the projection matrix.
   *
   * Uses random Gaussian projection with appropriate scaling.
   */
  initialize(): void {
    // Generate random projection matrix
    const matrix = new Float32Array(this.outputDim * this.inputDim);

    for (let i = 0; i < this.outputDim * this.inputDim; i++) {
      // Box-Muller for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

      // Scale by sqrt(1/outputDim) for JL property
      matrix[i] = z / Math.sqrt(this.outputDim);
    }

    this.projectionMatrix = matrix;
  }

  /**
   * Project a vector to lower dimension.
   *
   * @param vector - Input vector
   * @returns Projected vector
   */
  project(vector: Float32Array): Float32Array {
    if (!this.projectionMatrix) {
      this.initialize();
    }

    const result = new Float32Array(this.outputDim);

    for (let i = 0; i < this.outputDim; i++) {
      let sum = 0;
      for (let j = 0; j < this.inputDim; j++) {
        sum += this.projectionMatrix![i * this.inputDim + j] * vector[j];
      }
      result[i] = sum;
    }

    return result;
  }

  /**
   * Quantize projected vector.
   *
   * @param vector - Projected vector
   * @param bits - Bits for quantization (1-4)
   * @returns Quantized indices
   */
  quantize(vector: Float32Array, bits: number): Uint8Array {
    const levels = 1 << bits; // 2^bits levels
    const quantizer = new LloydMaxQuantizer(levels);
    const codebook = quantizer.generateCodebook(vector);
    return quantizer.quantize(vector, codebook);
  }

  /**
   * Compute approximate inner product using projected vectors.
   *
   * @param a - First vector
   * @param b - Second vector
   * @returns Approximate inner product
   */
  approximateInnerProduct(a: Float32Array, b: Float32Array): number {
    const projA = this.project(a);
    const projB = this.project(b);

    let dot = 0;
    for (let i = 0; i < projA.length; i++) {
      dot += projA[i] * projB[i];
    }

    // Scale back by output dimension to approximate original inner product
    return dot * this.outputDim;
  }
}

/**
 * TurboQuant key compressor.
 *
 * Implements 3-bit key compression using orthogonal rotation
 * and Lloyd-Max quantization.
 */
export class TurboQuantKeyCompressor {
  private dimension: number;
  private rotation: OrthogonalRotation | null = null;
  private codebook: Float32Array | null = null;

  /**
   * Create a key compressor.
   *
   * @param dimension - Key vector dimension
   */
  constructor(dimension: number) {
    this.dimension = dimension;
  }

  /**
   * Initialize the compressor with training data.
   *
   * @param trainingData - Sample keys for training
   */
  initialize(trainingData: Float32Array[]): void {
    // Generate rotation matrix
    this.rotation = new OrthogonalRotation(this.dimension);

    // Rotate training data
    const matrix = this.rotation.generate();
    const rotatedData = new Float32Array(trainingData.length * this.dimension);

    for (let i = 0; i < trainingData.length; i++) {
      const rotated = this.rotation.apply(matrix, trainingData[i]);
      rotatedData.set(rotated, i * this.dimension);
    }

    // Generate codebook for 3-bit quantization (8 levels)
    const quantizer = new LloydMaxQuantizer(8);
    this.codebook = quantizer.generateCodebook(rotatedData);
  }

  /**
   * Compress a key vector to 3-bit representation.
   *
   * @param key - Key vector
   * @returns Compressed representation
   */
  compress(key: Float32Array): { indices: Uint8Array; rotation: Float32Array } {
    if (!this.rotation || !this.codebook) {
      throw new Error('Compressor not initialized');
    }

    const matrix = this.rotation.generate();
    const rotated = this.rotation.apply(matrix, key);

    const quantizer = new LloydMaxQuantizer(8);
    const indices = quantizer.quantize(rotated, this.codebook);

    return { indices, rotation: matrix };
  }

  /**
   * Decompress key from 3-bit representation.
   *
   * @param compressed - Compressed key
   * @returns Decompressed key
   */
  decompress(compressed: { indices: Uint8Array; rotation: Float32Array }): Float32Array {
    if (!this.codebook) {
      throw new Error('Compressor not initialized');
    }

    const quantizer = new LloydMaxQuantizer(8);
    const rotated = quantizer.dequantize(compressed.indices, this.codebook);

    // Apply inverse rotation (R^T = R^-1 for orthogonal matrices)
    const result = new Float32Array(this.dimension);
    const rotation = compressed.rotation;

    for (let i = 0; i < this.dimension; i++) {
      let sum = 0;
      for (let j = 0; j < this.dimension; j++) {
        // Use transpose for inverse
        sum += rotation[j * this.dimension + i] * rotated[j];
      }
      result[i] = sum;
    }

    return result;
  }
}

/**
 * Group value quantizer for 2-bit value compression.
 *
 * Groups multiple value vectors and applies shared quantization.
 */
export class GroupValueQuantizer {
  private _groupSize: number;
  private codebook: Float32Array | null = null;

  /**
   * Create a group value quantizer.
   *
   * @param groupSize - Number of value vectors to group
   */
  constructor(groupSize = 4) {
    this._groupSize = groupSize;
  }

  /**
   * Get the group size for this quantizer.
   *
   * @returns The number of value vectors grouped together
   */
  get groupSize(): number {
    return this._groupSize;
  }

  /**
   * Initialize codebook from training data.
   *
   * @param trainingData - Sample values for training
   */
  initialize(trainingData: Float32Array[]): void {
    // Flatten training data
    const flatData = new Float32Array(trainingData.length * trainingData[0].length);
    for (let i = 0; i < trainingData.length; i++) {
      flatData.set(trainingData[i], i * trainingData[0].length);
    }

    // Generate codebook for 2-bit quantization (4 levels)
    const quantizer = new LloydMaxQuantizer(4);
    this.codebook = quantizer.generateCodebook(flatData);
  }

  /**
   * Compress value vectors using 2-bit quantization.
   *
   * @param values - Array of value vectors
   * @returns Compressed representation
   */
  compress(values: Float32Array[]): Uint8Array[] {
    if (!this.codebook) {
      throw new Error('Quantizer not initialized');
    }

    const quantizer = new LloydMaxQuantizer(4);
    return values.map(value => quantizer.quantize(value, this.codebook!));
  }

  /**
   * Decompress value vectors.
   *
   * @param compressed - Compressed indices
   * @returns Decompressed values
   */
  decompress(compressed: Uint8Array[]): Float32Array[] {
    if (!this.codebook) {
      throw new Error('Quantizer not initialized');
    }

    const quantizer = new LloydMaxQuantizer(4);
    return compressed.map(indices => quantizer.dequantize(indices, this.codebook!));
  }
}

/**
 * Main KV cache quantizer that combines key and value compression.
 */
export class KVCacheQuantizer {
  private keyCompressor: TurboQuantKeyCompressor | null = null;
  private valueQuantizer: GroupValueQuantizer | null = null;
  private dimension: number;

  /**
   * Create a KV cache quantizer.
   *
   * @param dimension - Vector dimension
   */
  constructor(dimension: number) {
    this.dimension = dimension;
  }

  /**
   * Initialize the quantizer.
   *
   * @param keyData - Sample keys for training
   * @param valueData - Sample values for training
   */
  initialize(keyData: Float32Array[], valueData: Float32Array[]): void {
    this.keyCompressor = new TurboQuantKeyCompressor(this.dimension);
    this.keyCompressor.initialize(keyData);

    this.valueQuantizer = new GroupValueQuantizer();
    this.valueQuantizer.initialize(valueData);
  }

  /**
   * Get compression statistics.
   *
   * @returns Compression statistics
   */
  getStats(): {
    keyCompression: number;
    valueCompression: number;
    totalCompression: number;
  } {
    const keyCompression = 16 / 3; // FP16 to 3-bit
    const valueCompression = 16 / 2; // FP16 to 2-bit
    const totalCompression = (keyCompression + valueCompression) / 2;

    return {
      keyCompression,
      valueCompression,
      totalCompression,
    };
  }
}
