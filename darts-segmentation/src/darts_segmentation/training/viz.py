"""Visualization utilities for the training module."""

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch


def plot_sample(x, y, y_pred, input_combinations: list[str]):
    """Plot a single sample with the input, the ground truth and the prediction.

    Args:
        x (torch.Tensor): The input tensor [C, H, W] (float).
        y (torch.Tensor): The ground truth tensor [H, W] (int).
        y_pred (torch.Tensor): The prediction tensor [H, W] (float).
        input_combinations (list[str]): The combinations of the input bands.

    Returns:
        tuple[Figure, dict[str, Axes]]: The figure and the axes of the plot.

    """
    x = x.cpu()
    y = y.cpu()
    y_pred = y_pred.detach().cpu()

    classification_labels = (y_pred > 0.5).int() + y * 2
    classification_labels = classification_labels.where(classification_labels != 0, torch.nan)

    # Calculate accuracy and iou
    true_positive = (classification_labels == 3).sum()
    false_positive = (classification_labels == 1).sum()
    false_negative = (classification_labels == 2).sum()
    acc = true_positive / (true_positive + false_positive + false_negative)

    cmap = mcolors.ListedColormap(["#cd43b2", "#3e0f2f", "#6cd875"])
    fig, axs = plt.subplot_mosaic([["a", "a", "b", "c"], ["a", "a", "d", "e"]], layout="constrained", figsize=(16, 8))

    # RGB Plot
    red_band = input_combinations.index("red")
    green_band = input_combinations.index("green")
    blue_band = input_combinations.index("blue")
    rgb = x[[red_band, green_band, blue_band]].transpose(0, 2).transpose(0, 1)
    ax_rgb = axs["a"]
    ax_rgb.imshow(rgb ** (1 / 1.4))
    ax_rgb.imshow(classification_labels, alpha=0.6, cmap=cmap, vmin=1, vmax=3)
    # Add a legend
    patches = [
        mpatches.Patch(color="#6cd875", label="True Positive"),
        mpatches.Patch(color="#3e0f2f", label="False Negative"),
        mpatches.Patch(color="#cd43b2", label="False Positive"),
    ]
    ax_rgb.legend(handles=patches, loc="upper left")
    # disable axis
    ax_rgb.axis("off")
    ax_rgb.set_title(f"Accuracy: {acc:.1%}")

    # NIR Plot
    nir_band = input_combinations.index("nir")
    nir = x[nir_band]
    ax_nir = axs["b"]
    ax_nir.imshow(nir, vmin=0, vmax=1)
    ax_nir.axis("off")
    ax_nir.set_title("NIR")

    # TCVIS Plot
    tcb_band = input_combinations.index("tc_brightness")
    tcg_band = input_combinations.index("tc_greenness")
    tcw_band = input_combinations.index("tc_wetness")
    tcvis = x[[tcb_band, tcg_band, tcw_band]].transpose(0, 2).transpose(0, 1)
    ax_tcv = axs["c"]
    ax_tcv.imshow(tcvis)
    ax_tcv.axis("off")
    ax_tcv.set_title("TCVIS")

    # NDVI Plot
    ndvi_band = input_combinations.index("ndvi")
    ndvi = x[ndvi_band]
    ax_ndvi = axs["d"]
    ax_ndvi.imshow(ndvi, vmin=0, vmax=1)
    ax_ndvi.axis("off")
    ax_ndvi.set_title("NDVI")

    # Prediction Plot
    ax_mask = axs["e"]
    ax_mask.imshow(y_pred, vmin=0, vmax=1)
    ax_mask.axis("off")
    ax_mask.set_title("Prediction")

    return fig, axs
