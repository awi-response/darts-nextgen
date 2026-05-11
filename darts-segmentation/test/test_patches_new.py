"""Tests for the new inference API using Patcher and forward.

These tests mirror the behaviour of the old, deprecated helpers and ensure
the refactored API produces identical results.
"""

import math

import pytest
import torch

from darts_segmentation.inference import Patcher, forward, patch_coords

# test_sizes = [10, 23, 60, 2000, 10008]
# test_patch_sizes = [8, 64, 256, 1024]
# test_overlaps = [0, 1, 3, 16, 64, 256]

# DEV
test_sizes = [10, 23, 60]
test_patch_sizes = [8, 64]
test_overlaps = [0, 1, 3, 16]


class DummyModel(torch.nn.Module):
    """A simple model that multiplies input by 2, for testing purposes."""

    def forward(self, x):
        return 2 * x


@pytest.mark.parametrize("size", test_sizes)
@pytest.mark.parametrize("patch_size", test_patch_sizes)
@pytest.mark.parametrize("overlap", test_overlaps)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_patch_prediction_new(size: int, patch_size: int, overlap: int, device: str):
    """Test prediction via `forward` + `Patcher` matches expected sigmoid(2*x).

    Uses a mock model that multiplies input by 2 (like the old tests).
    """
    if not size > patch_size > overlap:
        pytest.skip("unsupported configuration")

    model = DummyModel()

    h, w = size, size
    tensor_tiles = torch.rand((3, 1, h, w))
    patcher = Patcher(patch_size=patch_size, overlap=overlap)
    prediction = forward(
        tensor_tiles,
        model,
        patcher,
        batch_size=2,
        device=torch.device(device),
        reflection=0,
    )
    prediction_true = torch.sigmoid(2 * tensor_tiles).squeeze(1)
    assert prediction.shape == (3, h, w)
    torch.testing.assert_close(prediction, prediction_true)


@pytest.mark.parametrize("size", test_sizes)
@pytest.mark.parametrize("patch_size", test_patch_sizes)
@pytest.mark.parametrize("overlap", test_overlaps)
def test_create_patches_new(size: int, patch_size: int, overlap: int):
    """Tests creation of patches using `Patcher.deconstruct`.

    Verifies shapes and contents equal the corresponding regions of the input.
    """
    if not size > patch_size > overlap:
        pytest.skip("unsupported configuration")

    h, w = size, size
    tensor_tiles = torch.rand((3, 1, h, w))
    patcher = Patcher(patch_size=patch_size, overlap=overlap)
    patched = patcher.deconstruct(tensor_tiles)

    n_patches_h = math.ceil((h - overlap) / (patch_size - overlap))
    n_patches_w = math.ceil((w - overlap) / (patch_size - overlap))
    assert patched.patches.shape == (3, n_patches_h, n_patches_w, 1, patch_size, patch_size)

    step_size = patch_size - overlap
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            ipx = i * step_size
            jpx = j * step_size
            if ipx + patch_size > h:
                ipx = h - patch_size
            if jpx + patch_size > w:
                jpx = w - patch_size
            patch = patched.patches[:, i, j]
            true_patch = tensor_tiles[:, :, ipx : ipx + patch_size, jpx : jpx + patch_size]
            assert patch.shape == (3, 1, patch_size, patch_size)
            assert torch.allclose(patch, true_patch)


def test_patch_coords_example_generator():
    """Tests generation of the first 20 patch-coordinates for a 60x60 tile."""
    expected = [
        (0, (0, 0, 0, 0)),
        (1, (0, 5, 0, 1)),
        (2, (0, 10, 0, 2)),
        (3, (0, 15, 0, 3)),
        (4, (0, 20, 0, 4)),
        (5, (0, 25, 0, 5)),
        (6, (0, 30, 0, 6)),
        (7, (0, 35, 0, 7)),
        (8, (0, 40, 0, 8)),
        (9, (0, 45, 0, 9)),
        (10, (0, 50, 0, 10)),
        (11, (0, 52, 0, 11)),
        (12, (5, 0, 1, 0)),
        (13, (5, 5, 1, 1)),
        (14, (5, 10, 1, 2)),
        (15, (5, 15, 1, 3)),
        (16, (5, 20, 1, 4)),
        (17, (5, 25, 1, 5)),
        (18, (5, 30, 1, 6)),
        (19, (5, 35, 1, 7)),
    ]
    actual = list(enumerate(patch_coords(60, 60, 8, 3)))[:20]
    for expected_coords, actual_coords in zip(expected, actual):
        n_exp, (y_exp, x_exp, patch_idx_y_exp, patch_idx_x_exp) = expected_coords
        n_act, (y_act, x_act, patch_idx_y_act, patch_idx_x_act) = actual_coords

        assert n_exp == n_act
        assert y_exp == y_act
        assert x_exp == x_act
        assert patch_idx_y_exp == patch_idx_y_act
        assert patch_idx_x_exp == patch_idx_x_act


@pytest.mark.parametrize("size", test_sizes)
@pytest.mark.parametrize("patch_size", test_patch_sizes)
@pytest.mark.parametrize("overlap", test_overlaps)
def test_patch_coords_generator_logical(size: int, patch_size: int, overlap: int):
    """Logical checks for `patch_coords` across parameter sweeps.

    Ensures generated coordinates are within bounds and indices are consistent.
    """
    if not size > patch_size > overlap:
        pytest.skip("unsupported configuration")

    coords = list(enumerate(patch_coords(size, size, patch_size, overlap)))
    n_patches_h = math.ceil((size - overlap) / (patch_size - overlap))
    for n, (y, x, patch_idx_y, patch_idx_x) in coords:
        assert y >= 0
        assert x >= 0
        assert patch_idx_y >= 0
        assert patch_idx_x >= 0
        assert y + patch_size <= size
        assert x + patch_size <= size
        assert n == patch_idx_y * n_patches_h + patch_idx_x
