"""
Correctness tests comparing BoltEmbedding against scikit-learn.

These tests verify that BoltEmbedding returns the same nearest neighbors
as sklearn.neighbors.NearestNeighbors for all supported distance metrics.
"""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from boltembedding import BoltEmbedding, DistanceMetric, Device

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_embeddings(random_seed):
    """Small embedding dataset for quick tests."""
    return np.random.randn(100, 64).astype(np.float32)


@pytest.fixture
def medium_embeddings(random_seed):
    """Medium embedding dataset."""
    return np.random.randn(1000, 128).astype(np.float32)


@pytest.fixture
def single_query(random_seed):
    """Single query vector."""
    return np.random.randn(64).astype(np.float32)


@pytest.fixture
def batch_queries(random_seed):
    """Batch of query vectors."""
    return np.random.randn(10, 64).astype(np.float32)


# =============================================================================
# Euclidean Distance Tests
# =============================================================================


class TestEuclideanDistance:
    """Test Euclidean (L2) distance correctness."""

    def test_single_query_matches_sklearn(self, small_embeddings, single_query):
        """Single query should match sklearn NearestNeighbors."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        bolt.index(k)
        bolt_indices = bolt.query(single_query)

        # sklearn
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(small_embeddings)
        sklearn_indices = nn.kneighbors(
            single_query.reshape(1, -1), return_distance=False
        )[0]

        np.testing.assert_array_equal(bolt_indices, sklearn_indices)

    def test_batch_query_matches_sklearn(self, small_embeddings, batch_queries):
        """Batch queries should match sklearn NearestNeighbors."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        bolt.index(k)
        bolt_indices = bolt.query(batch_queries)

        # sklearn
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(small_embeddings)
        sklearn_indices = nn.kneighbors(batch_queries, return_distance=False)

        np.testing.assert_array_equal(bolt_indices, sklearn_indices)

    def test_distances_match_sklearn(self, small_embeddings, single_query):
        """Returned distances should match sklearn."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        bolt.index(k)
        bolt_indices, bolt_distances = bolt.query_with_distances(single_query)

        # sklearn returns actual L2 distances, we return squared
        # Convert our squared distances to actual L2 for comparison
        bolt_l2_distances = np.sqrt(bolt_distances)

        # sklearn
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(small_embeddings)
        sklearn_distances, sklearn_indices = nn.kneighbors(
            single_query.reshape(1, -1), return_distance=True
        )

        np.testing.assert_array_equal(bolt_indices, sklearn_indices[0])
        np.testing.assert_allclose(bolt_l2_distances, sklearn_distances[0], rtol=1e-5)


# =============================================================================
# Cosine Distance Tests
# =============================================================================


class TestCosineDistance:
    """Test Cosine distance correctness."""

    def test_single_query_matches_sklearn(self, small_embeddings, single_query):
        """Cosine distance should match sklearn."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings, metric=DistanceMetric.COSINE, device=Device.CPU
        )
        bolt.index(k)
        bolt_indices = bolt.query(single_query)

        # sklearn
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(small_embeddings)
        sklearn_indices = nn.kneighbors(
            single_query.reshape(1, -1), return_distance=False
        )[0]

        np.testing.assert_array_equal(bolt_indices, sklearn_indices)

    def test_batch_query_matches_sklearn(self, small_embeddings, batch_queries):
        """Batch cosine queries should match sklearn."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings, metric=DistanceMetric.COSINE, device=Device.CPU
        )
        bolt.index(k)
        bolt_indices = bolt.query(batch_queries)

        # sklearn
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(small_embeddings)
        sklearn_indices = nn.kneighbors(batch_queries, return_distance=False)

        np.testing.assert_array_equal(bolt_indices, sklearn_indices)

    def test_normalized_vectors(self, random_seed):
        """Test with pre-normalized vectors (common use case)."""
        k = 5
        embeddings = np.random.randn(100, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query = embeddings[0]  # Use first embedding as query

        bolt = BoltEmbedding(
            embeddings=embeddings, metric=DistanceMetric.COSINE, device=Device.CPU
        )
        bolt.index(k)
        bolt_indices = bolt.query(query)

        # First result should be the query itself (index 0)
        assert bolt_indices[0] == 0


# =============================================================================
# Manhattan Distance Tests
# =============================================================================


class TestManhattanDistance:
    """Test Manhattan (L1) distance correctness."""

    def test_single_query_matches_sklearn(self, small_embeddings, single_query):
        """Manhattan distance should match sklearn."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.MANHATTAN,
            device=Device.CPU,
        )
        bolt.index(k)
        bolt_indices = bolt.query(single_query)

        # sklearn
        nn = NearestNeighbors(n_neighbors=k, metric="manhattan")
        nn.fit(small_embeddings)
        sklearn_indices = nn.kneighbors(
            single_query.reshape(1, -1), return_distance=False
        )[0]

        np.testing.assert_array_equal(bolt_indices, sklearn_indices)

    def test_batch_query_matches_sklearn(self, small_embeddings, batch_queries):
        """Batch Manhattan queries should match sklearn."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.MANHATTAN,
            device=Device.CPU,
        )
        bolt.index(k)
        bolt_indices = bolt.query(batch_queries)

        # sklearn
        nn = NearestNeighbors(n_neighbors=k, metric="manhattan")
        nn.fit(small_embeddings)
        sklearn_indices = nn.kneighbors(batch_queries, return_distance=False)

        np.testing.assert_array_equal(bolt_indices, sklearn_indices)


# =============================================================================
# Inner Product Tests
# =============================================================================


class TestInnerProduct:
    """Test Inner Product (Maximum Inner Product Search) correctness."""

    def test_single_query_mips(self, small_embeddings, single_query):
        """Inner product should return highest dot products."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.INNER_PRODUCT,
            device=Device.CPU,
        )
        bolt.index(k)
        bolt_indices = bolt.query(single_query)

        # Manual computation: find indices with highest dot products
        dot_products = small_embeddings @ single_query
        expected_indices = np.argsort(dot_products)[-k:][::-1]

        np.testing.assert_array_equal(bolt_indices, expected_indices)

    def test_batch_query_mips(self, small_embeddings, batch_queries):
        """Batch inner product queries should return highest dot products."""
        k = 5

        # BoltEmbedding
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.INNER_PRODUCT,
            device=Device.CPU,
        )
        bolt.index(k)
        bolt_indices = bolt.query(batch_queries)

        # Manual computation
        for i, query in enumerate(batch_queries):
            dot_products = small_embeddings @ query
            expected_indices = np.argsort(dot_products)[-k:][::-1]
            np.testing.assert_array_equal(bolt_indices[i], expected_indices)


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_k_equals_1(self, small_embeddings, single_query):
        """k=1 should return single closest neighbor."""
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        bolt.index(k=1)
        result = bolt.query(single_query)

        assert result.shape == (1,)

    def test_k_equals_n(self, random_seed):
        """k=N should return all embeddings sorted by distance."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        query = np.random.randn(32).astype(np.float32)

        bolt = BoltEmbedding(
            embeddings=embeddings, metric=DistanceMetric.EUCLIDEAN, device=Device.CPU
        )
        bolt.index(k=10)
        result = bolt.query(query)

        assert result.shape == (10,)
        assert set(result) == set(range(10))

    def test_query_is_embedding(self, small_embeddings):
        """Querying with an embedding should return itself first."""
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        bolt.index(k=5)

        # Query with the 10th embedding
        query = small_embeddings[10]
        result = bolt.query(query)

        assert result[0] == 10

    def test_single_embedding(self, random_seed):
        """Should work with a single embedding."""
        embeddings = np.random.randn(1, 32).astype(np.float32)
        query = np.random.randn(32).astype(np.float32)

        bolt = BoltEmbedding(
            embeddings=embeddings, metric=DistanceMetric.EUCLIDEAN, device=Device.CPU
        )
        bolt.index(k=1)
        result = bolt.query(query)

        assert result[0] == 0

    def test_high_dimensional(self, random_seed):
        """Should work with high-dimensional embeddings."""
        embeddings = np.random.randn(50, 1024).astype(np.float32)
        query = np.random.randn(1024).astype(np.float32)

        bolt = BoltEmbedding(
            embeddings=embeddings, metric=DistanceMetric.EUCLIDEAN, device=Device.CPU
        )
        bolt.index(k=5)

        # Compare with sklearn
        nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        nn.fit(embeddings)
        sklearn_indices = nn.kneighbors(query.reshape(1, -1), return_distance=False)[0]

        np.testing.assert_array_equal(bolt.query(query), sklearn_indices)

    def test_different_dtypes(self, random_seed):
        """Should work with different float dtypes."""
        for dtype in [np.float16, np.float32, np.float64]:
            embeddings = np.random.randn(50, 32).astype(dtype)
            query = np.random.randn(32).astype(dtype)

            bolt = BoltEmbedding(
                embeddings=embeddings,
                metric=DistanceMetric.EUCLIDEAN,
                device=Device.CPU,
            )
            bolt.index(k=5)
            result = bolt.query(query)

            assert result.shape == (5,)


# =============================================================================
# Reindex Tests
# =============================================================================


class TestReindex:
    """Test reindex functionality."""

    def test_reindex_changes_k(self, small_embeddings, single_query):
        """Reindexing should change the number of results."""
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )

        bolt.index(k=5)
        result5 = bolt.query(single_query)
        assert result5.shape == (5,)

        bolt.reindex(k=10)
        result10 = bolt.query(single_query)
        assert result10.shape == (10,)

        # First 5 results should be the same
        np.testing.assert_array_equal(result5, result10[:5])


# =============================================================================
# Properties Tests
# =============================================================================


class TestProperties:
    """Test property accessors."""

    def test_shape_property(self, small_embeddings):
        """Shape should return correct dimensions."""
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        assert bolt.shape == (100, 64)

    def test_is_indexed_property(self, small_embeddings):
        """is_indexed should reflect indexing state."""
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        assert not bolt.is_indexed

        bolt.index(k=5)
        assert bolt.is_indexed

    def test_k_property(self, small_embeddings):
        """k property should return correct value."""
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        assert bolt.k is None

        bolt.index(k=7)
        assert bolt.k == 7

    def test_len(self, small_embeddings):
        """len() should return number of embeddings."""
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        assert len(bolt) == 100

    def test_repr(self, small_embeddings):
        """repr should return informative string."""
        bolt = BoltEmbedding(
            embeddings=small_embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        repr_str = repr(bolt)
        assert "BoltEmbedding" in repr_str
        assert "100" in repr_str
        assert "EUCLIDEAN" in repr_str
