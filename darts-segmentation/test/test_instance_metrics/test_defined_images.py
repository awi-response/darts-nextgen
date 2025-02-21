import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_test_image_square_offset_iou_80():
    """Create a test image with a single square in a corner with offset in prediction, resulting in a iou of 0.8.

    Metrics:
        Acc: 0.98
        IoU: 0.8
        F1: 0.98

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:40, 10:40] = 1
    preds[11:42, 11:43] = 1
    return preds, target


def create_test_image_blob_center_iou_1():
    """Create a test image with a single blob in the center.

    Metrics:
        Acc: 0.96
        IoU: 0.73
        F1: 0.85

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    s = 100
    x = torch.arange(0, s).float()
    y = torch.arange(0, s).float()
    xx, yy = torch.meshgrid(x, y)
    blob = torch.exp(-((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / (s / 4) ** 2)
    blob = blob / blob.max()
    target[:s, :s] = blob > 0.6
    preds[:s, :s] = blob
    return preds, target


def create_test_image_two_pred_blobs_single_target_blob():
    """Create a test image with a single blob in the center.

    First target blob is matched with first prediction blob, second prediction blob is false positive.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:50, 10:50] = 1
    target[70:80, 70:80] = 1
    preds[10:50, 10:50] = 0.7
    return preds, target


def create_test_image_false_positive():
    """Create a test image with no blobs in target but one blob in prediction.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    preds[10:50, 10:50] = 0.7
    return preds, target


def create_test_image_two_pred_blobs_two_target_blobs():
    """Create a test image with two blobs in target and two blobs in prediction.

    Both blobs match perfectly.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:50, 10:50] = 1
    target[70:80, 70:80] = 1
    preds[10:50, 10:50] = 0.7
    preds[70:80, 70:80] = 0.7
    return preds, target


def create_test_image_blob_in_corner():
    """Create a test image with a single blob in the corner.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[0:30, 0:30] = 1
    preds[0:30, 0:30] = 0.7
    return preds, target


def create_test_image_multiple_blobs():
    """Create a test image with multiple blobs scattered around.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:30, 10:30] = 1
    target[40:60, 40:60] = 1
    target[70:90, 70:90] = 1
    preds[10:30, 10:30] = 0.7
    preds[40:60, 40:60] = 0.7
    preds[70:90, 70:90] = 0.7
    return preds, target


def create_test_image_large_blob():
    """Create a test image with a large blob covering most of the image.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:90, 10:90] = 1
    preds[10:90, 10:90] = 0.7
    return preds, target


def create_test_image_small_blobs():
    """Create a test image with several small blobs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:20, 10:20] = 1
    target[30:40, 30:40] = 1
    target[50:60, 50:60] = 1
    target[70:80, 70:80] = 1
    preds[10:20, 10:20] = 0.7
    preds[30:40, 30:40] = 0.7
    preds[50:60, 50:60] = 0.7
    preds[70:80, 70:80] = 0.7
    return preds, target


def create_test_image_blob_touching_edges():
    """Create a test image with a blob touching the edges of the image.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[0:50, 0:100] = 1
    preds[0:50, 0:100] = 0.7
    return preds, target


def create_test_image_blob_with_hole():
    """Create a test image with a blob that has a hole in the middle.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:90, 10:90] = 1
    target[40:60, 40:60] = 0
    preds[10:90, 10:90] = 0.7
    preds[40:60, 40:60] = 0
    return preds, target


def create_test_image_diagonal_blobs():
    """Create a test image with blobs arranged diagonally.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:30, 10:30] = 1
    target[40:60, 40:60] = 1
    target[70:90, 70:90] = 1
    preds[10:30, 10:30] = 0.7
    preds[40:60, 40:60] = 0.7
    preds[70:90, 70:90] = 0.7
    return preds, target


def create_test_image_blob_with_noise():
    """Create a test image with a blob and random noise.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.rand(100, 100, dtype=torch.float32, device=DEVICE) * 0.2
    target[10:90, 10:90] = 1
    preds[10:90, 10:90] = 0.7
    return preds, target


def create_test_image_blob_with_partial_overlap():
    """Create a test image with a blob that partially overlaps with the target.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:50, 10:50] = 1
    preds[30:70, 30:70] = 0.7
    return preds, target


def create_test_image_blob_with_multiple_overlaps():
    """Create a test image with multiple overlapping blobs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:50, 10:50] = 1
    target[30:70, 30:70] = 1
    preds[20:60, 20:60] = 0.7
    return preds, target


def create_test_image_blob_with_different_shapes():
    """Create a test image with blobs of different shapes.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:30, 10:50] = 1
    target[60:90, 60:80] = 1
    preds[10:30, 10:50] = 0.7
    preds[60:90, 60:80] = 0.7
    return preds, target


def create_test_image_blob_with_gradient():
    """Create a test image with a blob that has a gradient.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    for i in range(10, 90):
        target[i, 10:90] = 1
        preds[i, 10:90] = (i - 10) / 80
    return preds, target


def create_test_image_blob_with_noise_and_gradient():
    """Create a test image with a blob that has both noise and gradient.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.rand(100, 100, dtype=torch.float32, device=DEVICE) * 0.2
    for i in range(10, 90):
        target[i, 10:90] = 1
        preds[i, 10:90] = (i - 10) / 80 + 0.2
    return preds, target


def create_test_image_blob_with_holes_and_noise():
    """Create a test image with a blob that has holes and noise.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.rand(100, 100, dtype=torch.float32, device=DEVICE) * 0.2
    target[10:90, 10:90] = 1
    target[40:60, 40:60] = 0
    preds[10:90, 10:90] = 0.7
    preds[40:60, 40:60] = 0
    return preds, target


def create_test_image_blob_with_different_intensities():
    """Create a test image with blobs of different intensities.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:30, 10:30] = 1
    target[40:60, 40:60] = 1
    target[70:90, 70:90] = 1
    preds[10:30, 10:30] = 0.5
    preds[40:60, 40:60] = 0.7
    preds[70:90, 70:90] = 0.9
    return preds, target


def create_test_image_blob_with_partial_intensity():
    """Create a test image with a blob that has partial intensity.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:90, 10:90] = 1
    preds[10:50, 10:50] = 0.7
    preds[50:90, 50:90] = 0.3
    return preds, target


def create_test_image_blob_with_partial_overlap_and_noise():
    """Create a test image with a blob that partially overlaps with the target and has noise.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.rand(100, 100, dtype=torch.float32, device=DEVICE) * 0.2
    target[10:50, 10:50] = 1
    preds[30:70, 30:70] = 0.7
    return preds, target


def create_test_image_blob_with_multiple_shapes():
    """Create a test image with blobs of multiple shapes.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:30, 10:50] = 1
    target[60:90, 60:80] = 1
    target[40:60, 40:60] = 1
    preds[10:30, 10:50] = 0.7
    preds[60:90, 60:80] = 0.7
    preds[40:60, 40:60] = 0.7
    return preds, target


def create_test_image_blob_with_partial_overlap_and_gradient():
    """Create a test image with a blob that partially overlaps with the target and has a gradient.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    for i in range(10, 50):
        target[i, 10:50] = 1
        preds[i + 20, 30:70] = (i - 10) / 40
    return preds, target


def create_test_image_blob_with_partial_overlap_and_holes():
    """Create a test image with a blob that partially overlaps with the target and has holes.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.zeros(100, 100, dtype=torch.float32, device=DEVICE)
    target[10:50, 10:50] = 1
    target[30:40, 30:40] = 0
    preds[30:70, 30:70] = 0.7
    preds[50:60, 50:60] = 0
    return preds, target


def create_test_image_blob_with_partial_overlap_and_noise_and_gradient():
    """Create a test image with a blob that partially overlaps with the target, has noise, and a gradient.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The prediction and target tensors.

    """
    target = torch.zeros(100, 100, dtype=torch.int64, device=DEVICE)
    preds = torch.rand(100, 100, dtype=torch.float32, device=DEVICE) * 0.2
    for i in range(10, 50):
        target[i, 10:50] = 1
        preds[i + 20, 30:70] = (i - 10) / 40 + 0.2
    return preds, target


def create_test_images():
    target_pred_pairs = [
        create_test_image_square_offset_iou_80(),
        create_test_image_blob_center_iou_1(),
        create_test_image_two_pred_blobs_single_target_blob(),
        create_test_image_false_positive(),
        create_test_image_two_pred_blobs_two_target_blobs(),
        create_test_image_blob_in_corner(),
        create_test_image_multiple_blobs(),
        create_test_image_large_blob(),
        create_test_image_small_blobs(),
        create_test_image_blob_touching_edges(),
        create_test_image_blob_with_hole(),
        create_test_image_diagonal_blobs(),
        create_test_image_blob_with_noise(),
        create_test_image_blob_with_partial_overlap(),
        create_test_image_blob_with_multiple_overlaps(),
        create_test_image_blob_with_different_shapes(),
        create_test_image_blob_with_gradient(),
        create_test_image_blob_with_noise_and_gradient(),
        create_test_image_blob_with_holes_and_noise(),
        create_test_image_blob_with_different_intensities(),
        create_test_image_blob_with_partial_intensity(),
        create_test_image_blob_with_partial_overlap_and_noise(),
        create_test_image_blob_with_multiple_shapes(),
        create_test_image_blob_with_partial_overlap_and_gradient(),
        create_test_image_blob_with_partial_overlap_and_holes(),
        create_test_image_blob_with_partial_overlap_and_noise_and_gradient(),
    ]
    return target_pred_pairs
