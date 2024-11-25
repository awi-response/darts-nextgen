"""Utility functions for Darts Preprocessing."""

import gc

try:
    import cupy as cp
except ImportError:
    cp = None


def free_cuda():
    """Free the CUDA memory."""
    if cp is not None:
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
