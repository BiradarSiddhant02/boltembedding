"""
Tests for error handling and exception cases.
"""

import numpy as np
import pytest

from boltembedding import (
    BoltEmbedding,
    DistanceMetric,
    Device,
    NotIndexedError,
    InvalidEmbeddingsError,
    InvalidQueryError,
    ConfigurationError,
)

# =============================================================================
# Configuration Errors
# =============================================================================


class TestConfigurationErrors:
    """Test configuration validation."""

    def test_no_embeddings_or_path(self):
        """Should raise error when neither embeddings nor path provided."""
        with pytest.raises(ConfigurationError, match="Either 'embeddings' or 'path'"):
            BoltEmbedding(metric=DistanceMetric.EUCLIDEAN)

    def test_both_embeddings_and_path(self):
        """Should raise error when both embeddings and path provided."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ConfigurationError, match="Only one of"):
            BoltEmbedding(embeddings=embeddings, path="fake.npy")

    def test_invalid_k_zero(self):
        """Should raise error for k=0."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)

        with pytest.raises(ConfigurationError, match="must be positive"):
            bolt.index(k=0)

    def test_invalid_k_negative(self):
        """Should raise error for negative k."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)

        with pytest.raises(ConfigurationError, match="must be positive"):
            bolt.index(k=-5)

    def test_k_larger_than_n(self):
        """Should raise error when k > number of embeddings."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)

        with pytest.raises(ConfigurationError, match="cannot be greater"):
            bolt.index(k=20)

    def test_invalid_k_type(self):
        """Should raise error for non-integer k."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)

        with pytest.raises(ConfigurationError, match="must be an integer"):
            bolt.index(k=5.5)


# =============================================================================
# Invalid Embeddings Errors
# =============================================================================


class TestInvalidEmbeddingsErrors:
    """Test embeddings validation."""

    def test_non_array_embeddings(self):
        """Should raise error for non-array embeddings."""
        with pytest.raises(InvalidEmbeddingsError, match="must be a numpy array"):
            BoltEmbedding(embeddings=[[1, 2, 3], [4, 5, 6]])

    def test_1d_embeddings(self):
        """Should raise error for 1D embeddings."""
        embeddings = np.random.randn(100).astype(np.float32)
        with pytest.raises(InvalidEmbeddingsError, match="must be 2D"):
            BoltEmbedding(embeddings=embeddings)

    def test_3d_embeddings(self):
        """Should raise error for 3D embeddings."""
        embeddings = np.random.randn(10, 10, 10).astype(np.float32)
        with pytest.raises(InvalidEmbeddingsError, match="must be 2D"):
            BoltEmbedding(embeddings=embeddings)

    def test_empty_embeddings(self):
        """Should raise error for empty embeddings."""
        embeddings = np.zeros((0, 32), dtype=np.float32)
        with pytest.raises(InvalidEmbeddingsError, match="empty"):
            BoltEmbedding(embeddings=embeddings)

    def test_zero_dimension_embeddings(self):
        """Should raise error for zero-dimensional embeddings."""
        embeddings = np.zeros((10, 0), dtype=np.float32)
        with pytest.raises(InvalidEmbeddingsError, match="dimension is 0"):
            BoltEmbedding(embeddings=embeddings)

    def test_invalid_dtype(self):
        """Should raise error for invalid dtype."""
        embeddings = np.random.randint(0, 100, size=(10, 32))
        with pytest.raises(InvalidEmbeddingsError, match="dtype"):
            BoltEmbedding(embeddings=embeddings)

    def test_nan_embeddings(self):
        """Should raise error for NaN values."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        embeddings[5, 10] = np.nan
        with pytest.raises(InvalidEmbeddingsError, match="NaN"):
            BoltEmbedding(embeddings=embeddings)

    def test_inf_embeddings(self):
        """Should raise error for infinite values."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        embeddings[5, 10] = np.inf
        with pytest.raises(InvalidEmbeddingsError, match="infinite"):
            BoltEmbedding(embeddings=embeddings)

    def test_file_not_found(self):
        """Should raise error for non-existent file."""
        with pytest.raises(InvalidEmbeddingsError, match="not found"):
            BoltEmbedding(path="/nonexistent/path/embeddings.npy")


# =============================================================================
# Not Indexed Errors
# =============================================================================


class TestNotIndexedErrors:
    """Test errors when querying before indexing."""

    def test_query_before_index(self):
        """Should raise error when querying before index()."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)

        query = np.random.randn(32).astype(np.float32)
        with pytest.raises(NotIndexedError, match="Call index"):
            bolt.query(query)

    def test_query_with_distances_before_index(self):
        """Should raise error when querying with distances before index()."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)

        query = np.random.randn(32).astype(np.float32)
        with pytest.raises(NotIndexedError):
            bolt.query_with_distances(query)


# =============================================================================
# Invalid Query Errors
# =============================================================================


class TestInvalidQueryErrors:
    """Test query validation."""

    def test_wrong_dimension_query(self):
        """Should raise error for wrong query dimension."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)
        bolt.index(k=5)

        query = np.random.randn(64).astype(np.float32)  # Wrong dimension
        with pytest.raises(InvalidQueryError, match="dimension"):
            bolt.query(query)

    def test_3d_query(self):
        """Should raise error for 3D query."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)
        bolt.index(k=5)

        query = np.random.randn(5, 5, 32).astype(np.float32)
        with pytest.raises(InvalidQueryError, match="1D.*or 2D"):
            bolt.query(query)

    def test_nan_query(self):
        """Should raise error for NaN in query."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)
        bolt.index(k=5)

        query = np.random.randn(32).astype(np.float32)
        query[10] = np.nan
        with pytest.raises(InvalidQueryError, match="NaN"):
            bolt.query(query)

    def test_inf_query(self):
        """Should raise error for infinite values in query."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)
        bolt.index(k=5)

        query = np.random.randn(32).astype(np.float32)
        query[10] = np.inf
        with pytest.raises(InvalidQueryError, match="infinite"):
            bolt.query(query)

    def test_non_array_query(self):
        """Should raise error for non-array query."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        bolt = BoltEmbedding(embeddings=embeddings)
        bolt.index(k=5)

        with pytest.raises(InvalidQueryError, match="numpy or JAX array"):
            bolt.query([1, 2, 3])


# =============================================================================
# File Loading Errors
# =============================================================================


class TestFileLoadingErrors:
    """Test file loading error handling."""

    def test_invalid_npy_file(self, tmp_path):
        """Should raise error for invalid .npy file."""
        # Create an invalid file
        bad_file = tmp_path / "bad.npy"
        bad_file.write_text("not a numpy file")

        with pytest.raises(InvalidEmbeddingsError, match="Failed to load"):
            BoltEmbedding(path=str(bad_file))
