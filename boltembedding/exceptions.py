"""
Custom exceptions for BoltEmbedding.
"""


class BoltEmbeddingError(Exception):
    """Base exception for all BoltEmbedding errors."""

    pass


class NotIndexedError(BoltEmbeddingError):
    """Raised when query is attempted before indexing."""

    def __init__(
        self, message: str = "Index not built. Call index(k) before querying."
    ):
        super().__init__(message)


class InvalidEmbeddingsError(BoltEmbeddingError):
    """Raised when embeddings are invalid or malformed."""

    def __init__(self, message: str = "Invalid embeddings provided."):
        super().__init__(message)


class InvalidQueryError(BoltEmbeddingError):
    """Raised when query vector is invalid."""

    def __init__(self, message: str = "Invalid query vector provided."):
        super().__init__(message)


class DeviceError(BoltEmbeddingError):
    """Raised when there's an issue with device configuration."""

    def __init__(self, message: str = "Device configuration error."):
        super().__init__(message)


class ConfigurationError(BoltEmbeddingError):
    """Raised when there's an invalid configuration."""

    def __init__(self, message: str = "Invalid configuration."):
        super().__init__(message)
