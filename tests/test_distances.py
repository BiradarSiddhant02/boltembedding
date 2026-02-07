# Copyright 2026 Siddhant Biradar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for distance metric implementations.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from boltembedding.distances import (
    precompute_euclidean,
    euclidean_distance_single,
    precompute_cosine,
    cosine_distance_single,
    precompute_inner_product,
    inner_product_distance_single,
    precompute_manhattan,
    manhattan_distance_single,
    get_distance_functions,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def embeddings():
    """Sample embeddings."""
    np.random.seed(42)
    return jnp.array(np.random.randn(100, 64).astype(np.float32))


@pytest.fixture
def query():
    """Sample query."""
    np.random.seed(123)
    return jnp.array(np.random.randn(64).astype(np.float32))


# =============================================================================
# Euclidean Distance Tests
# =============================================================================


class TestEuclideanDistance:
    """Test Euclidean distance implementation."""

    def test_precompute_shape(self, embeddings):
        """Precomputed norms should have correct shape."""
        norms = precompute_euclidean(embeddings)
        assert norms.shape == (100,)

    def test_distance_shape(self, embeddings, query):
        """Distance output should have correct shape."""
        norms = precompute_euclidean(embeddings)
        distances = euclidean_distance_single(embeddings, norms, query)
        assert distances.shape == (100,)

    def test_distance_non_negative(self, embeddings, query):
        """Squared distances should be non-negative."""
        norms = precompute_euclidean(embeddings)
        distances = euclidean_distance_single(embeddings, norms, query)
        assert jnp.all(distances >= -1e-6)  # Allow small numerical error

    def test_self_distance_zero(self, embeddings):
        """Distance from embedding to itself should be ~0."""
        norms = precompute_euclidean(embeddings)
        query = embeddings[0]
        distances = euclidean_distance_single(embeddings, norms, query)
        assert jnp.abs(distances[0]) < 1e-5

    def test_matches_numpy(self, embeddings, query):
        """Should match numpy implementation."""
        norms = precompute_euclidean(embeddings)
        jax_distances = euclidean_distance_single(embeddings, norms, query)

        # NumPy reference
        np_embeddings = np.array(embeddings)
        np_query = np.array(query)
        np_distances = np.sum((np_embeddings - np_query) ** 2, axis=1)

        np.testing.assert_allclose(jax_distances, np_distances, rtol=1e-5)


# =============================================================================
# Cosine Distance Tests
# =============================================================================


class TestCosineDistance:
    """Test Cosine distance implementation."""

    def test_precompute_shape(self, embeddings):
        """Precomputed norms should have correct shape."""
        norms = precompute_cosine(embeddings)
        assert norms.shape == (100,)

    def test_precompute_positive(self, embeddings):
        """Norms should be positive."""
        norms = precompute_cosine(embeddings)
        assert jnp.all(norms > 0)

    def test_distance_range(self, embeddings, query):
        """Cosine distance should be in [0, 2]."""
        norms = precompute_cosine(embeddings)
        distances = cosine_distance_single(embeddings, norms, query)
        assert jnp.all(distances >= -1e-6)
        assert jnp.all(distances <= 2.0 + 1e-6)

    def test_self_distance_zero(self, embeddings):
        """Distance from embedding to itself should be ~0."""
        norms = precompute_cosine(embeddings)
        query = embeddings[0]
        distances = cosine_distance_single(embeddings, norms, query)
        assert jnp.abs(distances[0]) < 1e-5

    def test_orthogonal_distance(self):
        """Orthogonal vectors should have cosine distance of 1."""
        embeddings = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        query = jnp.array([1.0, 0.0])

        norms = precompute_cosine(embeddings)
        distances = cosine_distance_single(embeddings, norms, query)

        assert jnp.abs(distances[0]) < 1e-5  # Same direction
        assert jnp.abs(distances[1] - 1.0) < 1e-5  # Orthogonal

    def test_opposite_distance(self):
        """Opposite vectors should have cosine distance of 2."""
        embeddings = jnp.array([[1.0, 0.0], [-1.0, 0.0]])
        query = jnp.array([1.0, 0.0])

        norms = precompute_cosine(embeddings)
        distances = cosine_distance_single(embeddings, norms, query)

        assert jnp.abs(distances[0]) < 1e-5  # Same direction
        assert jnp.abs(distances[1] - 2.0) < 1e-5  # Opposite


# =============================================================================
# Inner Product Tests
# =============================================================================


class TestInnerProduct:
    """Test Inner Product distance implementation."""

    def test_precompute_shape(self, embeddings):
        """Precompute should return placeholder of correct shape."""
        result = precompute_inner_product(embeddings)
        assert result.shape == (100,)

    def test_distance_shape(self, embeddings, query):
        """Distance output should have correct shape."""
        norms = precompute_inner_product(embeddings)
        distances = inner_product_distance_single(embeddings, norms, query)
        assert distances.shape == (100,)

    def test_higher_similarity_lower_distance(self, embeddings):
        """Higher dot product should give lower (more negative) distance."""
        query = embeddings[0]
        scaled_embedding = embeddings[0] * 2  # Same direction, larger magnitude

        embeddings_test = jnp.vstack([embeddings[0:1], scaled_embedding.reshape(1, -1)])
        norms = precompute_inner_product(embeddings_test)
        distances = inner_product_distance_single(embeddings_test, norms, query)

        # Larger magnitude embedding should have more negative distance
        assert distances[1] < distances[0]

    def test_matches_negative_dot(self, embeddings, query):
        """Should match negative dot product."""
        norms = precompute_inner_product(embeddings)
        distances = inner_product_distance_single(embeddings, norms, query)

        expected = -jnp.dot(embeddings, query)
        np.testing.assert_allclose(distances, expected, rtol=1e-5)


# =============================================================================
# Manhattan Distance Tests
# =============================================================================


class TestManhattanDistance:
    """Test Manhattan (L1) distance implementation."""

    def test_precompute_shape(self, embeddings):
        """Precompute should return placeholder of correct shape."""
        result = precompute_manhattan(embeddings)
        assert result.shape == (100,)

    def test_distance_shape(self, embeddings, query):
        """Distance output should have correct shape."""
        norms = precompute_manhattan(embeddings)
        distances = manhattan_distance_single(embeddings, norms, query)
        assert distances.shape == (100,)

    def test_distance_non_negative(self, embeddings, query):
        """Manhattan distances should be non-negative."""
        norms = precompute_manhattan(embeddings)
        distances = manhattan_distance_single(embeddings, norms, query)
        assert jnp.all(distances >= 0)

    def test_self_distance_zero(self, embeddings):
        """Distance from embedding to itself should be 0."""
        norms = precompute_manhattan(embeddings)
        query = embeddings[0]
        distances = manhattan_distance_single(embeddings, norms, query)
        assert jnp.abs(distances[0]) < 1e-5

    def test_matches_numpy(self, embeddings, query):
        """Should match numpy implementation."""
        norms = precompute_manhattan(embeddings)
        jax_distances = manhattan_distance_single(embeddings, norms, query)

        # NumPy reference
        np_embeddings = np.array(embeddings)
        np_query = np.array(query)
        np_distances = np.sum(np.abs(np_embeddings - np_query), axis=1)

        np.testing.assert_allclose(jax_distances, np_distances, rtol=1e-5)

    def test_simple_case(self):
        """Test with known values."""
        embeddings = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        query = jnp.array([0.0, 0.0])

        norms = precompute_manhattan(embeddings)
        distances = manhattan_distance_single(embeddings, norms, query)

        expected = jnp.array([0.0, 2.0, 4.0])
        np.testing.assert_allclose(distances, expected, rtol=1e-5)


# =============================================================================
# Registry Tests
# =============================================================================


class TestDistanceRegistry:
    """Test distance function registry."""

    def test_get_euclidean(self):
        """Should return euclidean functions."""
        funcs = get_distance_functions("euclidean")
        assert "precompute" in funcs
        assert "distance" in funcs

    def test_get_cosine(self):
        """Should return cosine functions."""
        funcs = get_distance_functions("cosine")
        assert "precompute" in funcs
        assert "distance" in funcs

    def test_get_inner_product(self):
        """Should return inner product functions."""
        funcs = get_distance_functions("inner_product")
        assert "precompute" in funcs
        assert "distance" in funcs

    def test_get_manhattan(self):
        """Should return manhattan functions."""
        funcs = get_distance_functions("manhattan")
        assert "precompute" in funcs
        assert "distance" in funcs

    def test_invalid_metric(self):
        """Should raise error for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_distance_functions("unknown_metric")
