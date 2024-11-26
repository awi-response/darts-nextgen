"""Utility functions around cuda, e.g. memory management."""

import gc

import torch

try:
    import cupy as cp
except ImportError:
    cp = None


def free_cupy():
    """Free the CUDA memory of cupy."""
    if cp is not None:
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def free_torch():
    """Free the CUDA memory of pytorch."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
