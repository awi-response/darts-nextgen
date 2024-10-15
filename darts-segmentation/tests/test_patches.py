"""Tests for the utility functions used for patched and stacked prediction."""

import math

import pytest
import torch

from darts_segmentation.utils import create_patches, patch_coords, predict_in_patches


@pytest.mark.parametrize("size", [10, 60, 500, 2000])
@pytest.mark.parametrize("patch_size", [8, 64, 1024])
@pytest.mark.parametrize("overlap", [0, 1, 3, 8, 16, 64])
def test_patch_prediction(size: int, patch_size: int, overlap: int):
    """Tests the prediction function with a mock model (*2) and a random tensor."""
    # Skip tests for invalid parameter to be able to to larger sweeps
    if not size > patch_size > overlap:
        return

    def model(x):
        return 2 * x

    h, w = size, size
    tensor_tiles = torch.rand((3, 1, h, w))
    prediction = predict_in_patches(model, tensor_tiles, patch_size=patch_size, overlap=overlap)
    prediction_true = torch.sigmoid(2 * tensor_tiles).squeeze(1)
    assert prediction.shape == (3, h, w)
    torch.testing.assert_allclose(prediction, prediction_true)


@pytest.mark.parametrize("size", [10, 60, 500, 2000])
@pytest.mark.parametrize("patch_size", [8, 64, 1024])
@pytest.mark.parametrize("overlap", [0, 1, 3, 8, 16, 64])
def test_create_patches(size: int, patch_size: int, overlap: int):
    """Tests the creation of patches."""
    # Skip tests for invalid parameter to be able to to larger sweeps
    if not size > patch_size > overlap:
        return

    h, w = size, size
    tensor_tiles = torch.rand((3, 1, h, w))
    patches = create_patches(tensor_tiles, patch_size=patch_size, overlap=overlap)
    n_patches_h = math.ceil(h / (patch_size - overlap))
    n_patches_w = math.ceil(h / (patch_size - overlap))
    assert patches.shape == (3, n_patches_h, n_patches_w, 1, patch_size, patch_size)

    step_size = patch_size - overlap
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            ipx = i * step_size
            jpx = j * step_size
            if ipx + patch_size > h:
                ipx = h - patch_size
            if jpx + patch_size > w:
                jpx = w - patch_size
            patch = patches[:, i, j]
            true_patch = tensor_tiles[:, :, ipx : ipx + patch_size, jpx : jpx + patch_size]
            assert patch.shape == (3, 1, patch_size, patch_size)
            assert torch.allclose(patch, true_patch)


def test_patch_coords_example_generator():
    """Tests the generation of the generation of patch-coordinates.

    Tests the first 20 patch-coordinates for a tile fo size 60x60px with a patch-size of 8 and an overlap of 3.
    """
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


@pytest.mark.parametrize("size", [10, 60, 500, 2000])
@pytest.mark.parametrize("patch_size", [8, 64, 1024])
@pytest.mark.parametrize("overlap", [0, 1, 3, 8, 16, 64])
def test_patch_coords_generator_logical(size: int, patch_size: int, overlap: int):
    """Tests the generation of the generation of patch-coordinates.

    Tests the first 20 patch-coordinates for a tile fo size 60x60px with a patch-size of 8 and an overlap of 3.
    """
    # Skip tests for invalid parameter to be able to to larger sweeps
    if not size > patch_size > overlap:
        return

    coords = list(enumerate(patch_coords(size, size, patch_size, overlap)))
    n_patches_h = math.ceil(size / (patch_size - overlap))
    for n, (y, x, patch_idx_y, patch_idx_x) in coords:
        assert y >= 0
        assert x >= 0
        assert patch_idx_y >= 0
        assert patch_idx_x >= 0
        assert y + patch_size <= size
        assert x + patch_size <= size
        assert n == patch_idx_y * n_patches_h + patch_idx_x
