# BoltEmbedding

A production-ready, high-performance nearest neighbor embedding search library powered by JAX.

## Features

- **JAX-Accelerated**: Leverages JAX for high-performance vectorized operations
- **Multi-Device Support**: CPU, CUDA (GPU), and TPU backends
- **Multiple Distance Metrics**: Euclidean (L2), Cosine Similarity, Inner Product, Manhattan (L1)
- **Batch Optimized**: Efficient batch query processing with VMAP
- **Production Ready**: Comprehensive error handling and input validation
- **Memory Flexible**: Supports both in-memory arrays and file-based loading

## Installation

```bash
pip install boltembedding

# For CUDA support
pip install boltembedding[cuda]

# For TPU support
pip install boltembedding[tpu]

# For development (includes test and benchmark dependencies)
pip install boltembedding[dev]
```

## Quick Start

```python
import numpy as np
from boltembedding import BoltEmbedding, DistanceMetric, Device

# Create random embeddings
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Initialize with in-memory embeddings
bolt = BoltEmbedding(
    embeddings=embeddings,
    metric=DistanceMetric.EUCLIDEAN,
    device=Device.CPU
)

# Build the index
bolt.index(k=10)

# Query single vector
query = np.random.randn(768).astype(np.float32)
indices = bolt.query(query)

# Batch query
queries = np.random.randn(100, 768).astype(np.float32)
batch_indices = bolt.query(queries)
```

## Loading from File

```python
from boltembedding import BoltEmbedding, DistanceMetric, Device

bolt = BoltEmbedding(
    path="embeddings.npy",
    metric=DistanceMetric.COSINE,
    device=Device.CUDA
)
bolt.index(k=5)
```

## Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `EUCLIDEAN` | L2 distance (default) | General purpose |
| `COSINE` | Cosine similarity | Text embeddings, normalized vectors |
| `INNER_PRODUCT` | Dot product | Maximum inner product search |
| `MANHATTAN` | L1 distance | Sparse data, robust to outliers |

## Device Selection

```python
from boltembedding import Device

Device.CPU    # Force CPU execution
Device.CUDA   # Use NVIDIA GPU
Device.TPU    # Use Google TPU
```

## API Reference

### `BoltEmbedding`

#### Constructor

```python
BoltEmbedding(
    embeddings: np.ndarray | None = None,
    path: str | None = None,
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    device: Device = Device.CPU
)
```

**Parameters:**
- `embeddings`: In-memory numpy array of shape (N, D)
- `path`: Path to .npy file containing embeddings
- `metric`: Distance metric to use
- `device`: Compute device

*Note: Either `embeddings` or `path` must be provided, but not both.*

#### Methods

- `index(k: int) -> None`: Build the index for k-nearest neighbor search
- `query(query: np.ndarray) -> np.ndarray`: Find k nearest neighbors
- `query_with_distances(query: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: Return indices and distances

#### Properties

- `shape`: Tuple of (num_embeddings, embedding_dim)
- `is_indexed`: Whether the index has been built
- `k`: Current k value (None if not indexed)

## Running Tests

```bash
# Install test dependencies
pip install boltembedding[test]

# Run tests
pytest tests/
```

## Running Benchmarks

```bash
# Install benchmark dependencies
pip install boltembedding[benchmark]

# Run benchmarks
python benchmarks/benchmark.py
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
