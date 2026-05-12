"""Tests for the new inference API using Patcher and forward.

These tests mirror the behaviour of the old, deprecated helpers and ensure
the refactored API produces identical results.
"""

import math

import pytest
import torch

from darts_segmentation.inference import (
    Patcher,
    _forward,
    _forward_on_device,
    _forward_streaming,
    _gen_batches,
    forward,
    patch_coords,
)

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
@pytest.mark.parametrize("tensor_device", ["cpu", "cuda"])
@pytest.mark.parametrize("model_device", ["cpu", "cuda"])
def test_patch_prediction_new(size: int, patch_size: int, overlap: int, tensor_device: str, model_device: str):
    """Test prediction via `forward` + `Patcher` matches expected sigmoid(2*x).

    Uses a mock model that multiplies input by 2 (like the old tests).
    """
    if not size > patch_size > overlap:
        pytest.skip("unsupported configuration")
    if "cuda" in (tensor_device, model_device):
        assert torch.cuda.is_available(), "CUDA is required for this test run"

    model = DummyModel()

    h, w = size, size
    tensor_tiles = torch.rand((3, 1, h, w), device=torch.device(tensor_device))
    patcher = Patcher(patch_size=patch_size, overlap=overlap)
    prediction = forward(
        tensor_tiles,
        model,
        patcher,
        batch_size=2,
        device=torch.device(model_device),
        reflection=0,
    )
    prediction_true = torch.sigmoid(2 * tensor_tiles).squeeze(1)
    assert prediction.shape == (3, h, w)
    torch.testing.assert_close(prediction, prediction_true)


def _make_complex_pattern(height: int, width: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    ramp = (yy.float() / max(1, height - 1)) + (xx.float() / max(1, width - 1))
    checker = ((yy + xx) % 2).float() * 0.25
    waves = 0.15 * torch.sin(yy.float() / 3.0) + 0.1 * torch.cos(xx.float() / 5.0)
    pattern = ramp + checker + waves
    return pattern


@pytest.mark.parametrize("size", test_sizes)
@pytest.mark.parametrize("patch_size", test_patch_sizes)
@pytest.mark.parametrize("overlap", test_overlaps)
def test_create_patches_new(size: int, patch_size: int, overlap: int):
    """Tests creation of patches using `Patcher.patchify`.

    Verifies shapes and contents equal the corresponding regions of the input.
    """
    if not size > patch_size > overlap:
        pytest.skip("unsupported configuration")

    h, w = size, size
    tensor_tiles = torch.rand((3, 1, h, w))
    patcher = Patcher(patch_size=patch_size, overlap=overlap)
    patched = patcher.patchify(tensor_tiles)

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


def test_reconstruct_complex_pattern_overlap():
    height, width = 31, 29
    patch_size, overlap = 9, 3
    pattern = _make_complex_pattern(height, width)
    tensor_tiles = pattern.unsqueeze(0).unsqueeze(0)
    patcher = Patcher(patch_size=patch_size, overlap=overlap)
    patched = patcher.patchify(tensor_tiles)
    probability_patches = patched.patches.squeeze(3)

    reconstruction = patcher.reconstruct(probability_patches, patched)
    assert reconstruction.shape == (1, height, width)

    weights = torch.zeros((height, width), dtype=reconstruction.dtype)
    soft_margin = patcher.soft_margin.cpu().squeeze(0)
    for pc in patched.patches_coordinates:
        yslice = slice(pc.y, pc.y + patch_size)
        xslice = slice(pc.x, pc.x + patch_size)
        weights[yslice, xslice] += soft_margin
    mask = weights > 0

    torch.testing.assert_close(reconstruction[0][mask], pattern[mask])
    assert torch.allclose(reconstruction[0][~mask], torch.zeros_like(reconstruction[0][~mask]))


def test_forward_reflection_padding_matches_pointwise_model():
    height, width = 25, 27
    patch_size, overlap = 11, 3
    pattern = _make_complex_pattern(height, width)
    tensor_tiles = pattern.unsqueeze(0).unsqueeze(0)
    patcher = Patcher(patch_size=patch_size, overlap=overlap)

    prediction = forward(
        tensor_tiles,
        DummyModel(),
        patcher,
        batch_size=4,
        device=torch.device("cpu"),
        reflection=2,
    )
    expected = torch.sigmoid(2 * tensor_tiles).squeeze(1)
    torch.testing.assert_close(prediction, expected)


def test_gen_batches_nan_handling():
    patch_size = 6
    patches = torch.rand((4, 1, patch_size, patch_size))
    patches[:2] = torch.nan
    patches[2, :, 0, 0] = torch.nan
    patches[3, :, 1, 1] = torch.nan

    batches = list(_gen_batches(patches, batch_size=2))
    assert len(batches) == 1
    batch = batches[0]
    assert batch.bslice == slice(2, 4)
    assert torch.isnan(batch.data).sum() == 0
    assert batch.data.shape == (2, 1, patch_size, patch_size)


def test_forward_same_device_matches_reference():
    patch_size = 7
    patches = torch.rand((5, 1, patch_size, patch_size))
    model = DummyModel()
    predicted = _forward(patches, model, batch_size=2)
    expected = torch.sigmoid(model(patches)).squeeze(1)
    torch.testing.assert_close(predicted, expected)


def test_forward_on_device_matches_reference():
    assert torch.cuda.is_available(), "CUDA is required for this test run"
    patch_size = 7
    patches = torch.rand((6, 1, patch_size, patch_size))
    device = torch.device("cuda")
    model = DummyModel().to(device)
    predicted = _forward_on_device(patches, model, batch_size=3, device=device)
    expected = torch.sigmoid(model(patches.to(device))).squeeze(1).cpu()
    torch.testing.assert_close(predicted.cpu(), expected)


def test_forward_streaming_matches_reference():
    assert torch.cuda.is_available(), "CUDA is required for this test run"
    patch_size = 9
    patches = torch.rand((5, 1, patch_size, patch_size)).pin_memory()
    device = torch.device("cuda")
    model = DummyModel().to(device)
    predicted = _forward_streaming(patches, model, batch_size=2, device=device)
    expected = torch.sigmoid(model(patches.to(device))).squeeze(1).cpu()
    torch.testing.assert_close(predicted.cpu(), expected)


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
