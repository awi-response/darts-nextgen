import torch

from darts_segmentation.metrics.boundary_helpers import erode_pytorch, get_boundary
from darts_segmentation.metrics.instance_helpers import mask_to_instances, match_instances

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_pytorch_erode():
    eroded_expected = torch.zeros(100, 100, dtype=torch.uint8, device=DEVICE)
    eroded_expected[12:41, 12:42] = 1
    preds = torch.zeros(100, 100, dtype=torch.uint8, device=DEVICE)
    preds[11:42, 11:43] = 1
    eroded = erode_pytorch(preds.unsqueeze(0), iterations=1).squeeze()
    assert torch.allclose(eroded, eroded_expected)


def test_get_boundary():
    img = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    img[10:50, 10:50] = 1
    boundary_expected = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    boundary_expected[10:50, 10:50] = 1
    boundary_expected[13:47, 13:47] = 0
    boundary = get_boundary(img.unsqueeze(0))
    assert torch.allclose(boundary, boundary_expected)


def test_mask_to_instances():
    img = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    img[10:40, 10:40] = 1
    img[50:80, 50:80] = 1
    img[10:40, 50:80] = 1
    expected_instances = torch.zeros(100, 100, dtype=torch.uint8, device=DEVICE)
    expected_instances[10:40, 10:40] = 1
    expected_instances[50:80, 50:80] = 3
    expected_instances[10:40, 50:80] = 2

    instances = mask_to_instances(img.unsqueeze(0))[0]
    assert torch.allclose(instances, expected_instances)


def test_match_instances():
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    # Match 1
    target[10:40, 10:40] = 1
    preds[11:42, 11:43] = 1
    # Match 2
    target[50:80, 50:80] = 1
    preds[51:82, 51:83] = 1
    # No-Match 1
    target[10:40, 50:80] = 1
    # No-Match 2
    preds[50:80, 10:40] = 1

    target_instances = mask_to_instances(target.unsqueeze(0))[0]
    preds_instances = mask_to_instances(preds.unsqueeze(0))[0]

    tp, fp, fn = match_instances(target_instances, preds_instances)
    assert tp == 2
    assert fp == 1
    assert fn == 1
