"""
Core BoltEmbedding implementation.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp

from boltembedding.types import DistanceMetric, Device
from boltembedding.exceptions import (
    NotIndexedError,
    InvalidEmbeddingsError,
    InvalidQueryError,
    DeviceError,
    ConfigurationError,
)
from boltembedding.distances import get_distance_functions


class BoltEmbedding:
    """
    High-performance nearest neighbor embedding search powered by JAX.

    Supports multiple distance metrics and compute devices (CPU, CUDA, TPU).
    Optimized for batch queries using JAX's VMAP.
    """

    # Valid embedding dtypes
    _VALID_DTYPES = (np.float16, np.float32, np.float64)

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        path: str | None = None,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        device: Device = Device.CPU,
    ):
        """
        Initialize BoltEmbedding.

        Args:
            embeddings: In-memory numpy array of shape (N, D). Mutually exclusive with path.
            path: Path to .npy file containing embeddings. Mutually exclusive with embeddings.
            metric: Distance metric to use for similarity search.
            device: Compute device (CPU, CUDA, or TPU).

        Raises:
            ConfigurationError: If neither or both embeddings and path are provided.
            InvalidEmbeddingsError: If embeddings are malformed.
            DeviceError: If requested device is not available.
        """
        # Validate input configuration
        self._validate_input_config(embeddings, path)

        # Store configuration
        self._path = path
        self._metric = metric
        self._device = device

        # Configure JAX device
        self._configure_device(device)

        # Load and validate embeddings
        self._embeddings_np = self._load_embeddings(embeddings, path)
        self._validate_embeddings(self._embeddings_np)

        # JAX arrays (populated on index())
        self._embeddings: jnp.ndarray | None = None
        self._precomputed_norms: jnp.ndarray | None = None

        # Index state
        self._k: int | None = None
        self._is_indexed: bool = False

        # Get distance functions for the metric
        metric_name = self._metric.name.lower()
        self._distance_funcs = get_distance_functions(metric_name)

        # Compiled query functions (set on index())
        self._query_single_fn = None
        self._query_batch_fn = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of embeddings (num_embeddings, embedding_dim)."""
        return self._embeddings_np.shape

    @property
    def is_indexed(self) -> bool:
        """Return whether the index has been built."""
        return self._is_indexed

    @property
    def k(self) -> int | None:
        """Return the current k value, or None if not indexed."""
        return self._k

    @property
    def metric(self) -> DistanceMetric:
        """Return the distance metric."""
        return self._metric

    @property
    def device(self) -> Device:
        """Return the compute device."""
        return self._device

    # =========================================================================
    # Public Methods
    # =========================================================================

    def index(self, k: int) -> None:
        """
        Build the index for k-nearest neighbor search.

        This transfers embeddings to the compute device and precomputes
        any required norms for efficient distance computation.

        Args:
            k: Number of nearest neighbors to return in queries.

        Raises:
            ConfigurationError: If k is invalid.
        """
        # Validate k
        self._validate_k(k)
        self._k = k

        # Transfer embeddings to device
        self._embeddings = jax.device_put(self._embeddings_np)

        # Precompute norms based on metric
        precompute_fn = self._distance_funcs["precompute"]
        self._precomputed_norms = precompute_fn(self._embeddings)

        # Build JIT-compiled query functions
        self._build_query_functions()

        self._is_indexed = True

    def query(self, query: np.ndarray) -> np.ndarray:
        """
        Find the k nearest neighbors for the given query vector(s).

        Args:
            query: Query vector of shape (D,) or batch of queries (M, D).

        Returns:
            Indices of k nearest neighbors. Shape (k,) for single query,
            or (M, k) for batch queries.

        Raises:
            NotIndexedError: If index() has not been called.
            InvalidQueryError: If query shape is incompatible.
        """
        self._check_indexed()
        query = self._validate_and_prepare_query(query)

        indices = self._query_batch_fn(
            self._precomputed_norms, self._embeddings, self._k, query
        )

        # Block until computation is complete
        indices.block_until_ready()

        # Convert to numpy and handle single query case
        indices_np = np.asarray(indices)

        if indices_np.shape[0] == 1:
            return indices_np[0]
        return indices_np

    def query_with_distances(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the k nearest neighbors with their distances.

        Args:
            query: Query vector of shape (D,) or batch of queries (M, D).

        Returns:
            Tuple of (indices, distances). Shapes match query() output.

        Raises:
            NotIndexedError: If index() has not been called.
            InvalidQueryError: If query shape is incompatible.
        """
        self._check_indexed()
        query = self._validate_and_prepare_query(query)

        indices, distances = self._query_batch_with_distances_fn(
            self._precomputed_norms, self._embeddings, self._k, query
        )

        # Block until computation is complete
        indices.block_until_ready()
        distances.block_until_ready()

        # Convert to numpy
        indices_np = np.asarray(indices)
        distances_np = np.asarray(distances)

        if indices_np.shape[0] == 1:
            return indices_np[0], distances_np[0]
        return indices_np, distances_np

    def reindex(self, k: int) -> None:
        """
        Rebuild the index with a new k value.

        More efficient than creating a new instance as embeddings
        are already on device.

        Args:
            k: New number of nearest neighbors.
        """
        self._validate_k(k)
        self._k = k
        self._build_query_functions()

    # =========================================================================
    # Private Methods - Validation
    # =========================================================================

    def _validate_input_config(
        self, embeddings: np.ndarray | None, path: str | None
    ) -> None:
        """Validate that exactly one of embeddings or path is provided."""
        if embeddings is None and path is None:
            raise ConfigurationError("Either 'embeddings' or 'path' must be provided.")
        if embeddings is not None and path is not None:
            raise ConfigurationError(
                "Only one of 'embeddings' or 'path' can be provided, not both."
            )

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """Validate embeddings array."""
        if not isinstance(embeddings, np.ndarray):
            raise InvalidEmbeddingsError(
                f"Embeddings must be a numpy array, got {type(embeddings).__name__}."
            )

        if embeddings.ndim != 2:
            raise InvalidEmbeddingsError(
                f"Embeddings must be 2D (N, D), got shape {embeddings.shape}."
            )

        if embeddings.shape[0] == 0:
            raise InvalidEmbeddingsError("Embeddings array is empty.")

        if embeddings.shape[1] == 0:
            raise InvalidEmbeddingsError("Embedding dimension is 0.")

        if embeddings.dtype not in self._VALID_DTYPES:
            raise InvalidEmbeddingsError(
                f"Embeddings dtype must be one of {self._VALID_DTYPES}, "
                f"got {embeddings.dtype}."
            )

        if not np.isfinite(embeddings).all():
            raise InvalidEmbeddingsError("Embeddings contain NaN or infinite values.")

    def _validate_k(self, k: int) -> None:
        """Validate k parameter."""
        if not isinstance(k, int):
            raise ConfigurationError(f"k must be an integer, got {type(k).__name__}.")

        if k <= 0:
            raise ConfigurationError(f"k must be positive, got {k}.")

        n_embeddings = self._embeddings_np.shape[0]
        if k > n_embeddings:
            raise ConfigurationError(
                f"k ({k}) cannot be greater than number of embeddings ({n_embeddings})."
            )

    def _validate_and_prepare_query(self, query: np.ndarray) -> jnp.ndarray:
        """Validate query and convert to proper shape."""
        if not isinstance(query, (np.ndarray, jnp.ndarray)):
            raise InvalidQueryError(
                f"Query must be a numpy or JAX array, got {type(query).__name__}."
            )

        query = np.asarray(query)

        if query.ndim == 1:
            query = query.reshape(1, -1)
        elif query.ndim != 2:
            raise InvalidQueryError(
                f"Query must be 1D (D,) or 2D (M, D), got shape {query.shape}."
            )

        expected_dim = self._embeddings_np.shape[1]
        if query.shape[1] != expected_dim:
            raise InvalidQueryError(
                f"Query dimension {query.shape[1]} does not match "
                f"embedding dimension {expected_dim}."
            )

        if not np.isfinite(query).all():
            raise InvalidQueryError("Query contains NaN or infinite values.")

        return jax.device_put(query)

    def _check_indexed(self) -> None:
        """Check if index has been built."""
        if not self._is_indexed:
            raise NotIndexedError()

    # =========================================================================
    # Private Methods - Device Configuration
    # =========================================================================

    def _configure_device(self, device: Device) -> None:
        """Configure JAX to use the specified device."""
        device_str = device.value

        try:
            available_devices = jax.devices()
            device_types = {d.platform for d in available_devices}

            # Map our device enum to JAX platform names
            platform_map = {
                "cpu": "cpu",
                "cuda": "gpu",  # JAX calls CUDA devices "gpu"
                "tpu": "tpu",
            }

            jax_platform = platform_map.get(device_str, device_str)

            if jax_platform not in device_types:
                available = ", ".join(sorted(device_types))
                raise DeviceError(
                    f"Requested device '{device_str}' is not available. "
                    f"Available devices: {available}."
                )

            # Set JAX to use only this platform
            jax.config.update("jax_platforms", jax_platform)

        except Exception as e:
            if isinstance(e, DeviceError):
                raise
            raise DeviceError(f"Failed to configure device '{device_str}': {e}")

    # =========================================================================
    # Private Methods - Loading
    # =========================================================================

    def _load_embeddings(
        self, embeddings: np.ndarray | None, path: str | None
    ) -> np.ndarray:
        """Load embeddings from memory or file."""
        if embeddings is not None:
            # Validation of type and dtype happens in _validate_embeddings
            return embeddings

        # Load from file
        if not os.path.exists(path):
            raise InvalidEmbeddingsError(f"Embeddings file not found: {path}")

        try:
            return np.load(path, mmap_mode="r")
        except Exception as e:
            raise InvalidEmbeddingsError(f"Failed to load embeddings from {path}: {e}")

    # =========================================================================
    # Private Methods - Query Function Building
    # =========================================================================

    def _build_query_functions(self) -> None:
        """Build JIT-compiled query functions."""
        distance_fn = self._distance_funcs["distance"]

        # Single query function
        def query_single(x_norm, embeddings, k, query):
            distances = distance_fn(embeddings, x_norm, query)
            # jax.lax.top_k returns largest, we want smallest -> negate
            _, topk_idx = jax.lax.top_k(-distances, k)
            return topk_idx

        # Single query with distances
        def query_single_with_distances(x_norm, embeddings, k, query):
            distances = distance_fn(embeddings, x_norm, query)
            neg_distances, topk_idx = jax.lax.top_k(-distances, k)
            return topk_idx, -neg_distances

        # VMAP for batch processing
        vmapped_query = jax.vmap(query_single, in_axes=(None, None, None, 0))
        vmapped_query_with_dist = jax.vmap(
            query_single_with_distances, in_axes=(None, None, None, 0)
        )

        # JIT compile with k as static argument
        self._query_batch_fn = jax.jit(vmapped_query, static_argnums=(2,))
        self._query_batch_with_distances_fn = jax.jit(
            vmapped_query_with_dist, static_argnums=(2,)
        )

        # Warm up JIT compilation
        dummy_query = jnp.zeros((1, self._embeddings.shape[1]))
        _ = self._query_batch_fn(
            self._precomputed_norms, self._embeddings, self._k, dummy_query
        )

    # =========================================================================
    # Special Methods
    # =========================================================================

    def __repr__(self) -> str:
        status = "indexed" if self._is_indexed else "not indexed"
        return (
            f"BoltEmbedding(shape={self.shape}, metric={self._metric.name}, "
            f"device={self._device.name}, k={self._k}, status={status})"
        )

    def __len__(self) -> int:
        return self._embeddings_np.shape[0]
