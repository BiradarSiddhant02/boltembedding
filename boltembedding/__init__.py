"""
BoltEmbedding - High-performance nearest neighbor embedding search powered by JAX.
"""

from boltembedding.core import BoltEmbedding
from boltembedding.types import DistanceMetric, Device
from boltembedding.exceptions import (
    BoltEmbeddingError,
    NotIndexedError,
    InvalidEmbeddingsError,
    InvalidQueryError,
    DeviceError,
    ConfigurationError,
)

__version__ = "1.0.0"
__all__ = [
    "BoltEmbedding",
    "DistanceMetric",
    "Device",
    "BoltEmbeddingError",
    "NotIndexedError",
    "InvalidEmbeddingsError",
    "InvalidQueryError",
    "DeviceError",
    "ConfigurationError",
]
