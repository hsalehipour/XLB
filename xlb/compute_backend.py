"""
Compute-backend enumeration for XLB.
"""

from enum import Enum, auto


class ComputeBackend(Enum):
    """Available compute backends.

    ``JAX``  — single-res, multi-GPU/TPU via JAX.
    ``WARP`` — single-res, single-GPU CUDA via NVIDIA Warp.
    ``NEON`` — single-res and multi-res, single-GPU CUDA via Neon (uses Warp kernels internally).
    """

    JAX = auto()
    WARP = auto()
    NEON = auto()
