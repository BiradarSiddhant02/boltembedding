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
Distance metric implementations using JAX.

All distance functions are designed to work with:
- embeddings: (N, D) array of database embeddings
- query: (D,) single query vector
- Precomputed norms where applicable

Returns distances where LOWER values indicate MORE similar items.
This allows consistent use of jax.lax.top_k with negation.
"""

import jax
import jax.numpy as jnp

# =============================================================================
# Euclidean (L2) Distance
# =============================================================================


def precompute_euclidean(embeddings: jnp.ndarray) -> jnp.ndarray:
    """Precompute squared L2 norms for embeddings."""
    return jnp.sum(embeddings**2, axis=1)


def euclidean_distance_single(
    embeddings: jnp.ndarray, x_norm: jnp.ndarray, query: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute squared Euclidean distance between query and all embeddings.

    ||x - q||^2 = ||x||^2 + ||q||^2 - 2 * x · q
    """
    q_norm = jnp.sum(query**2)
    distances = x_norm + q_norm - 2.0 * jnp.dot(embeddings, query)
    return distances


# =============================================================================
# Cosine Distance
# =============================================================================


def precompute_cosine(embeddings: jnp.ndarray) -> jnp.ndarray:
    """Precompute L2 norms for cosine similarity."""
    norms = jnp.sqrt(jnp.sum(embeddings**2, axis=1))
    # Avoid division by zero
    return jnp.maximum(norms, 1e-10)


def cosine_distance_single(
    embeddings: jnp.ndarray, x_norm: jnp.ndarray, query: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute cosine distance: 1 - cosine_similarity.

    cosine_similarity = (x · q) / (||x|| * ||q||)
    cosine_distance = 1 - cosine_similarity
    """
    q_norm = jnp.sqrt(jnp.sum(query**2))
    q_norm = jnp.maximum(q_norm, 1e-10)

    dot_products = jnp.dot(embeddings, query)
    cosine_sim = dot_products / (x_norm * q_norm)

    # Clamp to [-1, 1] to handle numerical errors
    cosine_sim = jnp.clip(cosine_sim, -1.0, 1.0)

    return 1.0 - cosine_sim


# =============================================================================
# Inner Product (Negative for top-k)
# =============================================================================


def precompute_inner_product(embeddings: jnp.ndarray) -> jnp.ndarray:
    """No precomputation needed for inner product. Returns placeholder."""
    return jnp.zeros(embeddings.shape[0])


def inner_product_distance_single(
    embeddings: jnp.ndarray,
    x_norm: jnp.ndarray,  # Unused, kept for consistent API
    query: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute negative inner product (for maximum inner product search).

    We negate so that higher similarity = lower "distance".
    """
    return -jnp.dot(embeddings, query)


# =============================================================================
# Manhattan (L1) Distance
# =============================================================================


def precompute_manhattan(embeddings: jnp.ndarray) -> jnp.ndarray:
    """No efficient precomputation for Manhattan. Returns placeholder."""
    return jnp.zeros(embeddings.shape[0])


def manhattan_distance_single(
    embeddings: jnp.ndarray,
    x_norm: jnp.ndarray,  # Unused, kept for consistent API
    query: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Manhattan (L1) distance: sum(|x - q|).
    """
    return jnp.sum(jnp.abs(embeddings - query), axis=1)


# =============================================================================
# Distance Function Registry
# =============================================================================

DISTANCE_FUNCTIONS = {
    "euclidean": {
        "precompute": precompute_euclidean,
        "distance": euclidean_distance_single,
    },
    "cosine": {
        "precompute": precompute_cosine,
        "distance": cosine_distance_single,
    },
    "inner_product": {
        "precompute": precompute_inner_product,
        "distance": inner_product_distance_single,
    },
    "manhattan": {
        "precompute": precompute_manhattan,
        "distance": manhattan_distance_single,
    },
}


def get_distance_functions(metric_name: str):
    """Get precompute and distance functions for a given metric."""
    if metric_name not in DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Available: {list(DISTANCE_FUNCTIONS.keys())}"
        )
    return DISTANCE_FUNCTIONS[metric_name]
