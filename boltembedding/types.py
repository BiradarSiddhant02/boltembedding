"""
Type definitions and enumerations for BoltEmbedding.
"""

from enum import Enum, auto


class DistanceMetric(Enum):
    """Supported distance metrics for nearest neighbor search."""

    EUCLIDEAN = auto()  # L2 distance: ||x - y||_2
    COSINE = auto()  # Cosine similarity: 1 - (x · y) / (||x|| * ||y||)
    INNER_PRODUCT = auto()  # Dot product: -x · y (negated for top-k)
    MANHATTAN = auto()  # L1 distance: ||x - y||_1


class Device(Enum):
    """Supported compute devices."""

    CPU = "cpu"
    CUDA = "cuda"
    TPU = "tpu"
