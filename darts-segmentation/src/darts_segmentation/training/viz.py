"""Visualization utilities for the training module."""

import itertools
import logging

import albumentations as A  # noqa: N812
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import ultraplot as uplt

from darts_segmentation.training.augmentations import Augmentation, get_augmentation

# TODO: New Plot: Threshold vs. F1-Score and IoU

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def plot_sample(
    x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor, band_names: list[str]
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot a single sample with the input, the ground truth and the prediction.

    This function does a few expections on the input:
    - The input is expected to be normalized to 0-1.
    - The prediction is expected to be converted from logits to prediction.
    - The target is expected to be a int or long tensor with values of:
        0 (negative class)
        1 (positive class) and
        2 (invalid pixels).

    Args:
        x (torch.Tensor): The input tensor [C, H, W] (float).
        y (torch.Tensor): The ground truth tensor [H, W] (int).
        y_pred (torch.Tensor): The prediction tensor [H, W] (float).
        band_names (list[str]): The combinations of the input bands.

    Returns:
        tuple[Figure, dict[str, Axes]]: The figure and the axes of the plot.

    """
    x = x.cpu()
    y = y.cpu()
    y_pred = y_pred.detach().cpu()

    # Make y class 2 invalids (replace 2 with nan)
    x = x.where(y != 2, torch.nan)
    y_pred = y_pred.where(y != 2, torch.nan)
    y = y.where(y != 2, torch.nan)

    # pred == 0, y == 0 -> 0 (true negative)
    # pred == 1, y == 0 -> 1 (false positive)
    # pred == 0, y == 1 -> 2 (false negative)
    # pred == 1, y == 1 -> 3 (true positive)
    classification_labels = (y_pred > 0.5).int() + y * 2
    classification_labels = classification_labels.where(classification_labels != 0, torch.nan)

    # Calculate f1 and iou
    true_positive = (classification_labels == 3).sum()
    false_positive = (classification_labels == 1).sum()
    false_negative = (classification_labels == 2).sum()
    true_negative = (classification_labels == 0).sum()
    acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    iou = true_positive / (true_positive + false_positive + false_negative)

    cmap = mcolors.ListedColormap(["#cd43b2", "#3e0f2f", "#6cd875"])
    fig, axs = plt.subplot_mosaic(
        # [["rgb", "rgb", "ndvi", "tcvis", "stats"], ["rgb", "rgb", "pred", "slope", "elev"]],
        [["rgb", "rgb", "pred", "tcvis"], ["rgb", "rgb", "ndvi", "slope"], ["none", "stats", "stats", "stats"]],
        # layout="constrained",
        figsize=(11, 8),
    )

    # Disable none plot
    axs["none"].axis("off")

    # RGB Plot
    ax_rgb = axs["rgb"]
    # disable axis
    ax_rgb.axis("off")
    is_rgb = "red" in band_names and "green" in band_names and "blue" in band_names
    if is_rgb:
        red_band = band_names.index("red")
        green_band = band_names.index("green")
        blue_band = band_names.index("blue")
        rgb = x[[red_band, green_band, blue_band]].transpose(0, 2).transpose(0, 1)
        ax_rgb.imshow(rgb ** (1 / 1.4))
        ax_rgb.set_title(f"Acc: {acc:.1%} F1: {f1:.1%} IoU: {iou:.1%}")
    else:
        # Plot empty with message that RGB is not provided
        ax_rgb.set_title("No RGB values are provided!")
    ax_rgb.imshow(classification_labels, alpha=0.6, cmap=cmap, vmin=1, vmax=3)
    # Add a legend
    patches = [
        mpatches.Patch(color="#6cd875", label="True Positive"),
        mpatches.Patch(color="#3e0f2f", label="False Negative"),
        mpatches.Patch(color="#cd43b2", label="False Positive"),
    ]
    ax_rgb.legend(handles=patches, loc="upper left")

    # NDVI Plot
    ax_ndvi = axs["ndvi"]
    ax_ndvi.axis("off")
    is_ndvi = "ndvi" in band_names
    if is_ndvi:
        ndvi_band = band_names.index("ndvi")
        ndvi = x[ndvi_band]
        ax_ndvi.imshow(ndvi, vmin=0, vmax=1, cmap="RdYlGn")
        ax_ndvi.set_title("NDVI")
    else:
        # Plot empty with message that NDVI is not provided
        ax_ndvi.set_title("No NDVI values are provided!")

    # TCVIS Plot
    ax_tcv = axs["tcvis"]
    ax_tcv.axis("off")
    is_tcvis = "tc_brightness" in band_names and "tc_greenness" in band_names and "tc_wetness" in band_names
    if is_tcvis:
        tcb_band = band_names.index("tc_brightness")
        tcg_band = band_names.index("tc_greenness")
        tcw_band = band_names.index("tc_wetness")
        tcvis = x[[tcb_band, tcg_band, tcw_band]].transpose(0, 2).transpose(0, 1)
        ax_tcv.imshow(tcvis)
        ax_tcv.set_title("TCVIS")
    else:
        ax_tcv.set_title("No TCVIS values are provided!")

    # Statistics Plot
    ax_stat = axs["stats"]
    if (y == 1).sum() > 0:
        n_bands = x.shape[0]
        n_pixel = x.shape[1] * x.shape[2]
        x_flat = x.flatten().cpu()
        y_flat = y.flatten().repeat(n_bands).cpu()
        bands = list(itertools.chain.from_iterable([band_names[i]] * n_pixel for i in range(n_bands)))
        plot_data = pd.DataFrame({"x": x_flat, "y": y_flat, "band": bands})
        if len(plot_data) > 50000:
            plot_data = plot_data.sample(50000)
        plot_data = plot_data.sort_values("band")
        sns.violinplot(
            x="x",
            y="band",
            hue="y",
            data=plot_data,
            split=True,
            inner="quart",
            fill=False,
            palette={1: "g", 0: ".35"},
            density_norm="width",
            ax=ax_stat,
        )
        ax_stat.set_title("Band Statistics")
    else:
        ax_stat.set_title("No positive labels in this sample!")
        ax_stat.axis("off")

    # Prediction Plot
    ax_mask = axs["pred"]
    ax_mask.imshow(y_pred, vmin=0, vmax=1)
    ax_mask.axis("off")
    ax_mask.set_title("Model Output")

    # Slope Plot
    ax_slope = axs["slope"]
    ax_slope.axis("off")
    is_slope = "slope" in band_names
    if is_slope:
        slope_band = band_names.index("slope")
        slope = x[slope_band]
        ax_slope.imshow(slope, cmap="cividis")
        # Add TPI as contour lines
        is_rel_elev = "relative_elevation" in band_names
        if is_rel_elev:
            rel_elev_band = band_names.index("relative_elevation")
            rel_elev = x[rel_elev_band]
            cs = ax_slope.contour(rel_elev, [0], colors="red", linewidths=0.3, alpha=0.6)
            ax_slope.clabel(cs, inline=True, fontsize=5, fmt="%.1f")

        ax_slope.set_title("Slope")
    else:
        # Plot empty with message that slope is not provided
        ax_slope.set_title("No Slope values are provided!")

    # Relative Elevation Plot
    # rel_elev_band = band_names.index("relative_elevation")
    # rel_elev = x[rel_elev_band]
    # ax_rel_elev = axs["elev"]
    # ax_rel_elev.imshow(rel_elev, cmap="cividis")
    # ax_rel_elev.axis("off")
    # ax_rel_elev.set_title("Relative Elevation")

    return fig, axs


def plot_augmentations(
    x: torch.Tensor, augmentations: list[Augmentation], band_names: list[str]
) -> tuple[uplt.Figure, uplt.gridspec.SubplotGrid]:
    """Plot augmentations applied to a sample image.

    Args:
        x (torch.Tensor): Input tensor [N, C, H, W] (float).
        augmentations (list[Augmentation]): List of augmentations to apply.
        band_names (list[str]): List of band names corresponding to the channels in x.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
        ultraplot.gridspec.SubplotGrid: The axes of the plot.

    """
    compose = get_augmentation(augmentations)
    augmentations: dict[str, A.BasicTransform] = {aug: get_augmentation([aug], True) for aug in augmentations}

    rgb_idx = [band_names.index(band) for band in ["red", "green", "blue"]]

    nrows = 1 + len(augmentations) + 4
    ncols = x.shape[0]
    fig, axs = uplt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 5, nrows * 5))
    for i in range(ncols):
        img = x[i, rgb_idx].permute(1, 2, 0).cpu().numpy()
        axs[0, i].imshow(img, vmin=0, vmax=0.1)
        axs[0, i].set_title("Original Image")
        for j, (aug_name, aug_fn) in enumerate(augmentations.items()):
            augmented = aug_fn(image=img)
            aug_img = augmented["image"]
            axs[j + 1, i].imshow(aug_img, vmin=0, vmax=0.1)
            axs[j + 1, i].set_title(f"Augmented: {aug_name}")

        # Apply full compose
        for j in range(4):
            augmented = compose(image=img)
            aug_img = augmented["image"]
            axs[j + 1 + len(augmentations), i].imshow(aug_img, vmin=0, vmax=0.1)
            axs[j + 1 + len(augmentations), i].set_title(f"Compose Augmentation {j + 1}")
    return fig, axs
