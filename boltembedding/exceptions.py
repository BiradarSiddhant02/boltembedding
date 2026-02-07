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
