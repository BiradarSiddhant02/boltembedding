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
