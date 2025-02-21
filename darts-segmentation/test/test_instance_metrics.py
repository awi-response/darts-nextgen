import torch

from darts_segmentation.metrics.binary_instance_prc import (
    BinaryInstanceAveragePrecision,
    BinaryInstancePrecisionRecallCurve,
)
from darts_segmentation.metrics.binary_instance_stat_scores import (
    BinaryInstanceAccuracy,
    BinaryInstanceConfusionMatrix,
    BinaryInstanceF1Score,
    BinaryInstancePrecision,
    BinaryInstanceRecall,
    _binary_stat_scores_tensor_validation,
)
from darts_segmentation.metrics.instance_helpers import mask_to_instances, match_instances

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_test_image(target, preds):
    import matplotlib.pyplot as plt
    from torchmetrics.functional.classification import accuracy, f1_score, jaccard_index

    bin_pred = (preds > 0.5).to(int)

    acc = accuracy(preds, target, task="binary")
    iou = jaccard_index(preds, target, task="binary")
    f1 = f1_score(preds, target, task="binary")

    target_size = (target.sum() / target.numel()).item()
    preds_size = (bin_pred.sum() / bin_pred.numel()).item()

    tp = (bin_pred * target).sum().item()
    fp = (bin_pred * (1 - target)).sum().item()
    fn = ((1 - bin_pred) * target).sum().item()

    tp_size = tp / target.numel()
    fp_size = fp / target.numel()
    fn_size = fn / target.numel()

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(target.squeeze().cpu().numpy(), cmap="gray")
    ax[0].set_title(f"Ground Truth\n{target_size:.2%} ({target.sum().item()})")
    ax[0].axis("off")

    ax[1].imshow(preds.squeeze().cpu().numpy(), cmap="gray")
    ax[1].set_title("Prediction")
    ax[1].axis("off")

    ax[2].imshow(bin_pred.squeeze().cpu().numpy(), cmap="gray")
    ax[2].set_title(f"Prediction (binarized)\n{preds_size:.2%} ({bin_pred.sum().item()})")
    ax[2].axis("off")

    ax[3].imshow(target.squeeze().cpu().numpy(), cmap="gray")
    ax[3].imshow(preds.squeeze().cpu().numpy(), cmap="jet", alpha=0.5)
    ax[3].set_title(f"Overlay\n{tp_size:.2%} TP, {fp_size:.2%} FP, {fn_size:.2%} FN\n({tp}, {fp}, {fn})")
    ax[3].axis("off")

    fig.suptitle(f"Acc: {acc:.2f} - IoU: {iou:.2f} - F1: {f1:.2f}")
    return fig


def create_test_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Simulate a batch of 4 images with 2 instances each, an instance is a random sizes blob on the image
    n, h, w = 8, 256, 256

    target = torch.zeros(n, h, w, dtype=torch.int64, device=device)
    preds = torch.zeros(n, h, w, dtype=torch.float32, device=device)
    # Image 1: A single square in a corner with offset in prediction
    target[0, 10:120, 10:120] = 1
    preds[0, 20:110, 5:125] = 0.8
    # Image 2: A single blob in the center
    s = 100
    x = torch.arange(0, s).float()
    y = torch.arange(0, s).float()
    xx, yy = torch.meshgrid(x, y)
    blob = torch.exp(-((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / (s / 4) ** 2)
    blob = blob / blob.max()
    target[1, 78 : 78 + s, 78 : 78 + s] = blob > 0.8
    preds[1, 70 : 70 + s, 70 : 70 + s] = blob
    # Image 3: Two blobs in target, one blob in prediction
    target[2, 10:110, 10:110] = 1
    target[2, 120:220, 120:220] = 1
    preds[2, 10:110, 10:110] = 0.7
    # Image 4: No blobs in target, one blob in prediction
    preds[3, 10:110, 10:110] = 0.7
    # Image 5: Two blobs in prediction, one blob in target
    target[4, 10:110, 10:110] = 1
    preds[4, 10:110, 10:110] = 0.7
    preds[4, 120:220, 120:220] = 0.7
    # Image 6: Two blobs in target, two blobs in prediction
    target[5, 10:110, 10:110] = 1
    target[5, 120:220, 120:220] = 1
    preds[5, 10:110, 10:110] = 0.7
    preds[5, 120:220, 120:220] = 0.7
    # Image 7: Two blobs in target, two blobs in prediction, one blob in prediction is false
    target[6, 10:110, 10:110] = 1
    target[6, 120:220, 120:220] = 1
    preds[6, 10:110, 10:110] = 0.7
    preds[6, 10:120, 120:220] = 0.7
    # Image 8: No blobs at all

    return preds, target


def test_binary_instance_recall():
    preds, target = create_test_images()
    metric = BinaryInstanceRecall()
    metric.update(preds, target)
    print(metric.compute())


def test_binary_instance_precision():
    preds, target = create_test_images()
    metric = BinaryInstancePrecision()
    metric.update(preds, target)
    print(metric.compute())


def test_binary_instance_f1_score():
    preds, target = create_test_images()
    metric = BinaryInstanceF1Score()
    metric.update(preds, target)
    print(metric.compute())


def test_binary_instance_accuracy():
    preds, target = create_test_images()
    metric = BinaryInstanceAccuracy()
    metric.update(preds, target)
    print(metric.compute())


def test_binary_instance_confusion_matrix():
    preds, target = create_test_images()
    metric = BinaryInstanceConfusionMatrix(normalize=True)
    metric.update(preds, target)
    print(metric.compute())


def test_binary_instance_prc():
    preds, target = create_test_images()
    metric = BinaryInstancePrecisionRecallCurve(thresholds=100)
    metric.update(preds, target)
    print(metric.compute())


def test_binary_instance_average_precision():
    preds, target = create_test_images()
    metric = BinaryInstanceAveragePrecision(thresholds=100)
    metric.update(preds, target)
    print(metric.compute())


def format_and_validate(preds, target, validate_args, multidim_average, ignore_index, threshold):
    if validate_args:
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
        if not preds.shape == target.shape:
            raise ValueError(
                f"Expected `preds` and `target` to have the same shape, but got {preds.shape} and {target.shape}."
            )
        if not preds.dim() == 3:
            raise ValueError(f"Expected `preds` and `target` to have 3 dimensions, but got {preds.dim()}.")

    # Format
    if preds.is_floating_point():
        if not torch.all((preds >= 0) * (preds <= 1)):
            # preds is logits, convert with sigmoid
            preds = preds.sigmoid()
        preds = preds > threshold

    if ignore_index is not None:
        idx = target == ignore_index
        target = target.clone()
        target[idx] = -1
    return preds, target


def test_stat_score_update():
    matching_threshold = 0.5
    threshold = 0.5

    preds, target = create_test_images()
    preds, target = format_and_validate(preds, target, True, "global", None, threshold)

    # Update state
    instance_list_target = mask_to_instances(target.to(torch.uint8))
    instance_list_preds = mask_to_instances(preds.to(torch.uint8))

    n = 8
    res = torch.zeros(n, 3)
    expected = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [2, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ]
    )
    for i, (target_i, preds_i) in enumerate(zip(instance_list_target, instance_list_preds)):
        tp, fp, fn = match_instances(target_i, preds_i, match_threshold=matching_threshold)
        res[i, :] = torch.tensor([tp, fp, fn])
        assert torch.all(res[i] == expected[i]), f"Expected [tp, fp, fn] {expected[i]} but got {res[i]}"
    print(res)
