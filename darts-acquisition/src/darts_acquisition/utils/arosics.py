"""Re-implementation of the AROSICS algorithm."""

import logging
from typing import Literal

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.fft import fft2, fftshift, ifft2

logger = logging.getLogger(__name__)


def _get_bands(
    target: xr.Dataset,
    reference: xr.Dataset,
    bands: list[str] | Literal["multiband"] | str = "multiband",
) -> list[str]:
    # Get a list of bands to align and validate them
    if isinstance(bands, str):
        if bands != "multiband":
            bands = [bands]
        else:
            # Use all bands that are in both datasets
            bands = list(set(target.data_vars) & set(reference.data_vars))
            if not bands:
                raise ValueError("No common bands found in target and reference datasets.")
    for band in bands:
        assert band in target.data_vars, f"Band '{band}' not found in target dataset."
        assert band in reference.data_vars, f"Band '{band}' not found in reference dataset."
        assert target[band].dtype == reference[band].dtype, (
            f"Band '{band}' has different dtype in target and reference datasets: "
            f"{target[band].dtype} vs {reference[band].dtype}."
        )
    return bands


def _validate_subset(
    subset: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | None = None,  # True: valid, False: invalid
    max_invalid_ratio: float = 0.01,
) -> bool:
    if isinstance(subset, xr.Dataset):
        for band in subset.variables:
            if not _validate_subset(subset[band], mask, max_invalid_ratio):
                return False
        return True
    # Check for NaN values
    if subset.isnull().any():
        return False
    # Check for mask
    if mask is not None:
        mask_invalid_ratio = 1 - mask.mean()
        if mask_invalid_ratio > max_invalid_ratio:
            return False
    return True


def _find_suitable_subset(
    target: xr.Dataset | xr.DataArray,
    reference: xr.Dataset | xr.DataArray,
    window_size: int = 256,
    target_mask: xr.DataArray | None = None,
    reference_mask: xr.DataArray | None = None,
    max_invalid_ratio: float = 0.01,
) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
    # Start naive with the center
    center_x = target.sizes["x"] // 2
    center_y = target.sizes["y"] // 2

    # Initialize direction vectors (right, bottom, left, top)
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    # Track current position, direction, turns and steps
    corner = (center_x - window_size // 2, center_y - window_size // 2)
    direction = 0
    steps_in_direction = 0
    turns_taken = 0

    for tries in range(1000):
        xslice = slice(corner[0], corner[0] + window_size)
        yslice = slice(corner[1], corner[1] + window_size)
        reference_subset = reference.isel(x=xslice, y=yslice)
        target_subset = target.isel(x=xslice, y=yslice)
        target_subset_mask = target_mask.isel(x=xslice, y=yslice) if target_mask is not None else None
        reference_subset_mask = reference_mask.isel(x=xslice, y=yslice) if reference_mask is not None else None
        # Check if the subset is valid
        is_valid = _validate_subset(
            target_subset,
            mask=target_subset_mask,
            max_invalid_ratio=max_invalid_ratio,
        ) and _validate_subset(
            reference_subset,
            mask=reference_subset_mask,
            max_invalid_ratio=max_invalid_ratio,
        )
        if is_valid:
            logger.debug(f"Found valid subset after {tries=} at corner {corner} with {direction=} and {window_size=}.")
            return target_subset, reference_subset
        # If not valid, shift the corner in a spiraling pattern
        corner = (corner[0] + dx[direction], corner[1] + dy[direction])

        # Check if we are still in bounds
        if (corner[0] < 0 or corner[0] + window_size > target.sizes["x"]) or (
            corner[1] < 0 or corner[1] + window_size > target.sizes["y"]
        ):
            logger.warning("Couldn't find a valid subset in the target and reference datasets.")
            break

        # Update direction if needed
        steps_in_direction += 1
        if steps_in_direction == turns_taken // 2 + 1:
            direction = (direction + 1) % 4
            steps_in_direction = 0
            turns_taken += 1
    return None, None


def _calculate_scps(reference: xr.DataArray, target: xr.DataArray) -> np.array:
    # Calculate the shifted cross power spectrum
    # This is a trick to avoid convoluted block matching:
    # Convolutions can be computed in the frequency domain with fourier transforms
    # Hence, we turn our images into the frequency domain, compute the cross power spectrum,
    # and then turn it back into the spatial domain.
    # The peak of the result tells us the spatial shift between the two images.
    ref_fft = fft2(reference.astype("complex64"))
    target_fft = fft2(target.astype("complex64"))
    eps = np.abs(ref_fft).max() * 1e-15
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_power_spectrum = (ref_fft * target_fft.conj()) / (abs(ref_fft) * abs(target_fft) + eps)
    cross_power_spectrum = ifft2(cross_power_spectrum)
    cross_power_spectrum = abs(cross_power_spectrum)
    shifted_cross_power_spectrum = fftshift(cross_power_spectrum)
    return shifted_cross_power_spectrum


def _calculate_offset(reference: xr.DataArray, target: xr.DataArray) -> tuple[int, int]:
    shifted_cross_power_spectrum = _calculate_scps(reference, target)
    # Find the peak in the cross power spectrum
    # The peak position in relation to the images center corresponds to the offset between the two images
    y_peak, x_peak = np.unravel_index(np.argmax(shifted_cross_power_spectrum), shifted_cross_power_spectrum.shape)
    x_offset = x_peak - reference.sizes["x"] // 2
    y_offset = y_peak - reference.sizes["y"] // 2
    return x_offset, y_offset


def get_offsets(  # noqa: C901
    target: xr.Dataset | xr.DataArray,
    reference: xr.Dataset | xr.DataArray,
    bands: list[str] | Literal["multiband"] | str = "multiband",
    subset: dict[Literal["x", "y"], slice] | Literal[False] | None = None,
    window_size: int = 256,
    target_mask: xr.DataArray | None = None,
    reference_mask: xr.DataArray | None = None,
    max_invalid_ratio: float = 0.01,
    resample_to: Literal["reference", "target"] | None = None,
) -> tuple[int, int]:
    """Get the offsets between a target and a reference using the AROSICS algorithm.

    Note:
        Assumes that the target and reference datasets have the same dimensions.

    Args:
        target (xr.Dataset | xr.DataArray): The target image dataset or dataarray to be aligned.
        reference (xr.Dataset | xr.DataArray): The reference image dataset or dataarray.
        bands (list[str] | Literal["multiband"] | str): The bands to use for alignment.
            Only used if the target and reference are datasets.
            If "multiband", all bands are used.
            This expects the target and reference datasets to have the same band names.
            If string, the respective band is used for alignment.
            If a list of strings, only the specified bands are used for alignment.
            Note: All bands are shifted by the same offset, even when using "multiband".
            With multiband, the offset is calculated from the average of all common and valid band offsets.
            This is slower but more robust than using a single band.
            If a band-specific offset is desired,
            please use the `get_dataarray_offsets` function for each band separately.
            Defaults to "multiband".
        subset (dict[Literal["x", "y"], slice] | Literal[False] | None): A dictionary of slices to use for alignment.
            If provided, only the specified subset of the target and reference datasets is used for alignment.
            The dictionary must contain the keys "x" and "y" with the respective slices.
            If False, the whole dataset is used for alignment.
            If None, will try to find a suitable subset automatically.
        window_size (int): The size of the window to use for alignment in case no subset is provided. Defaults to 256.
        target_mask (xr.DataArray | None): A mask for the target dataset.
            If provided, searches for a valid subset of the target dataset where the mask is True.
            If not provided, a mask is automatically inferred from the nodata value of the target dataset.
            Defaults to None.
        reference_mask (xr.DataArray | None): A mask for the reference dataset.
            If provided, searches for a valid subset of the reference dataset where the mask is True.
            If not provided, a mask is automatically inferred from the nodata value of the reference dataset.
            Defaults to None.
        max_invalid_ratio (float): The maximum ratio of invalid pixels in the target and reference subset masks.
            Is not used if the masks are not provided.
            Defaults to 0.01.
        resample_to (Literal["reference", "target"] | None): The dataset to resample the other dataset to.
            If "reference", the target dataset is resampled to the reference dataset.
            If "target", the reference dataset is resampled to the target dataset.
            Defaults to None.
            If None, no resampling is done.
            This assumes that the pixel grids of the target and reference datasets are already aligned.

    Returns:
        tuple[int, int]: The offsets in x and y direction between the target and reference datasets.

    Raises:
        ValueError: If the target and reference are not both datasets or dataarrays.
        ValueError: If no suitable subset is found for alignment.
            This can happen if the window_size is too large or if the masks are too restrictive.

    """
    # Check if both are datasets or dataarrays
    both_are_datasets = isinstance(target, xr.Dataset) and isinstance(reference, xr.Dataset)
    both_are_dataarrays = isinstance(target, xr.DataArray) and isinstance(reference, xr.DataArray)
    if not (both_are_datasets or both_are_dataarrays):
        raise ValueError("Both target and reference must be either xr.Dataset or xr.DataArray.")

    # Check if the dimentsions are x and y
    if "x" not in target.dims or "y" not in target.dims:
        raise ValueError("Target dataset must have dimensions 'x' and 'y'.")
    if "x" not in reference.dims or "y" not in reference.dims:
        raise ValueError("Reference dataset must have dimensions 'x' and 'y'.")

    if both_are_datasets:
        # Only work on subset of the bands
        bands = _get_bands(target, reference, bands)
        # Apply bands to the datasets
        target = target[bands]
        reference = reference[bands]

    # Check for matching crs
    if target.odc.geobox.crs != reference.odc.geobox.crs:
        logger.warning(
            f"Target and reference datasets have different CRS: {target.odc.geobox.crs} vs {reference.odc.geobox.crs}. "
            "Reprojecting reference to target CRS."
        )
        reference = reference.odc.reproject(target.odc.geobox.crs, resampling="nearest")
        reference_mask = (
            reference_mask.astype("int8").odc.reproject(target.odc.geobox.crs, resampling="nearest").astype("bool")
            if reference_mask is not None
            else None
        )

    # Get spatial intersection of the two images
    common_extent = reference.odc.geobox.geographic_extent.intersection(target.odc.geobox.geographic_extent)
    reference = reference.odc.crop(common_extent)
    target = target.odc.crop(common_extent)
    reference_mask = (
        reference_mask.odc.crop(common_extent).fillna(0.0).astype("bool") if reference_mask is not None else None
    )
    target_mask = target_mask.odc.crop(common_extent).fillna(0.0).astype("bool") if target_mask is not None else None

    # Resample if requested
    # We can savely use nearest resampling here, since the datatype will be converted to complex64 anyway
    # And the matching algorithm works in the frequency domain, so using we use the fastest resampling method
    if resample_to == "reference":
        target = target.odc.reproject(reference.odc.geobox, resampling="nearest")
        target_mask = (
            target_mask.astype("int8").odc.reproject(reference.odc.geobox, resampling="nearest").astype("bool")
            if target_mask is not None
            else None
        )
    elif resample_to == "target":
        reference = reference.odc.reproject(target.odc.geobox, resampling="nearest")
        reference_mask = (
            reference_mask.astype("int8").odc.reproject(target.odc.geobox, resampling="nearest").astype("bool")
            if reference_mask is not None
            else None
        )

    if subset is None:
        target, reference = _find_suitable_subset(
            target,
            reference,
            window_size=window_size,
            target_mask=target_mask,
            reference_mask=reference_mask,
            max_invalid_ratio=max_invalid_ratio,
        )
        if reference is None or target is None:
            raise ValueError(
                "No suitable subset found for alignment. Try decreasing the window size and check the masks."
            )
    elif isinstance(subset, dict):
        reference = reference.isel(x=subset["x"], y=subset["y"])
        target = target.isel(x=subset["x"], y=subset["y"])

    # Calculate the offset between the two subsets
    if both_are_dataarrays:
        x_offset, y_offset = _calculate_offset(reference, target)
        logger.debug(f"Offset: x_offset={x_offset}, y_offset={y_offset}")
        return x_offset, y_offset
    else:
        offsets = {
            "x": [],
            "y": [],
        }
        for band in bands:
            x_offset, y_offset = _calculate_offset(reference[band], target[band])
            offsets["x"].append(x_offset)
            offsets["y"].append(y_offset)

        # Calculate the average offset
        x_offset = np.mean(offsets["x"])
        y_offset = np.mean(offsets["y"])

        dbg_msg = f"Average offset: x_offset={x_offset}, y_offset={y_offset}"
        for i in range(len(bands)):
            dbg_msg += f"\n- Band {bands[i]}: x_offset={offsets['x'][i]}, y_offset={offsets['y'][i]}"
        logger.debug(dbg_msg)
        print(dbg_msg)
        return x_offset, y_offset


def align(
    target: xr.Dataset | xr.DataArray,
    reference: xr.Dataset | xr.DataArray,
    bands: list[str] | Literal["multiband"] | str = "multiband",
    subset: dict[Literal["x", "y"], slice] | Literal[False] | None = None,
    window_size: int = 256,
    target_mask: xr.DataArray | None = None,
    reference_mask: xr.DataArray | None = None,
    max_invalid_ratio: float = 0.01,
    resample_to: Literal["reference", "target"] | None = None,
    return_offsets: bool = False,
    inplace: bool = False,
) -> tuple[xr.Dataset | xr.DataArray, tuple[int, int]] | xr.Dataset | xr.DataArray:
    """Align a target to an reference using the AROSICS algorithm.

    Note:
        Assumes that the target and reference datasets have the same dimensions.

    Args:
        target (xr.Dataset | xr.DataArray): The target image dataset or dataarray to be aligned.
        reference (xr.Dataset | xr.DataArray): The reference image dataset or dataarray.
        bands (list[str] | Literal["multiband"] | str): The bands to use for alignment.
            Only used if the target and reference are datasets.
            If "multiband", all bands are used.
            This expects the target and reference datasets to have the same band names.
            If string, the respective band is used for alignment.
            If a list of strings, only the specified bands are used for alignment.
            Note: All bands are shifted by the same offset, even when using "multiband".
            With multiband, the offset is calculated from the average of all common and valid band offsets.
            This is slower but more robust than using a single band.
            If a band-specific offset is desired,
            please use the `get_dataarray_offsets` function for each band separately.
            Defaults to "multiband".
        subset (dict[Literal["x", "y"], slice] | Literal[False] | None): A dictionary of slices to use for alignment.
            If provided, only the specified subset of the target and reference datasets is used for alignment.
            The dictionary must contain the keys "x" and "y" with the respective slices.
            If False, the whole dataset is used for alignment.
            If None, will try to find a suitable subset automatically.
        window_size (int): The size of the window to use for alignment in case no subset is provided. Defaults to 256.
        target_mask (xr.DataArray | None): A mask for the target dataset.
            If provided, searches for a valid subset of the target dataset where the mask is True.
            If not provided, a mask is automatically inferred from the nodata value of the target dataset.
            Defaults to None.
        reference_mask (xr.DataArray | None): A mask for the reference dataset.
            If provided, searches for a valid subset of the reference dataset where the mask is True.
            If not provided, a mask is automatically inferred from the nodata value of the reference dataset.
            Defaults to None.
        max_invalid_ratio (float): The maximum ratio of invalid pixels in the target and reference subset masks.
            Is not used if the masks are not provided.
            Defaults to 0.01.
        resample_to (Literal["reference", "target"] | None): The dataset to resample the other dataset to.
            If "reference", the target dataset is resampled to the reference dataset.
            If "target", the reference dataset is resampled to the target dataset.
            Defaults to None.
            If None, no resampling is done.
            This assumes that the pixel grids of the target and reference datasets are already aligned.
        return_offsets (bool): If True, returns the offsets instead of aligning the target dataset.
        inplace (bool): If True, modifies the target dataset in place.

    Returns:
        tuple[xr.Dataset | xr.DataArray, tuple[int, int]] | xr.Dataset | xr.DataArray:
            The aligned target dataset or dataarray.
            If return_offsets is True, also returns the offsets in x and y direction as a tuple.

    """
    x_offset, y_offset = get_offsets(
        target,
        reference,
        bands=bands,
        subset=subset,
        window_size=window_size,
        target_mask=target_mask,
        reference_mask=reference_mask,
        max_invalid_ratio=max_invalid_ratio,
        resample_to=resample_to,
    )

    # Apply the offset to the original target dataset
    if not inplace:
        target = target.copy(deep=True)
    target["x"] = target.x + x_offset * target.odc.geobox.resolution.x
    target["y"] = target.y + y_offset * target.odc.geobox.resolution.y

    if return_offsets:
        return target, (x_offset, y_offset)
    else:
        return target


def visualize_alignment(  # noqa: C901
    target: xr.Dataset | xr.DataArray,
    reference: xr.Dataset | xr.DataArray,
    bands: list[str] | Literal["multiband"] | str = "multiband",
    subset: dict[Literal["x", "y"], slice] | Literal[False] | None = None,
    window_size: int = 256,
    target_mask: xr.DataArray | None = None,
    reference_mask: xr.DataArray | None = None,
    max_invalid_ratio: float = 0.01,
    resample_to: Literal["reference", "target"] | None = None,
) -> list[tuple[plt.Figure, list[plt.Axes]] | None]:
    """Visualized the alignment inner workings.

    Note:
        This function never works inplace!

    Args:
        target (xr.Dataset | xr.DataArray): The target image dataset or dataarray to be aligned.
        reference (xr.Dataset | xr.DataArray): The reference image dataset or dataarray.
        bands (list[str] | Literal["multiband"] | str): The bands to use for alignment.
            Only used if the target and reference are datasets.
            If "multiband", all bands are used.
            This expects the target and reference datasets to have the same band names.
            If string, the respective band is used for alignment.
            If a list of strings, only the specified bands are used for alignment.
            Note: All bands are shifted by the same offset, even when using "multiband".
            With multiband, the offset is calculated from the average of all common and valid band offsets.
            This is slower but more robust than using a single band.
            If a band-specific offset is desired,
            please use the `get_dataarray_offsets` function for each band separately.
            Defaults to "multiband".
        subset (dict[Literal["x", "y"], slice] | Literal[False] | None): A dictionary of slices to use for alignment.
            If provided, only the specified subset of the target and reference datasets is used for alignment.
            The dictionary must contain the keys "x" and "y" with the respective slices.
            If False, the whole dataset is used for alignment.
            If None, will try to find a suitable subset automatically.
        window_size (int): The size of the window to use for alignment in case no subset is provided. Defaults to 256.
        target_mask (xr.DataArray | None): A mask for the target dataset.
            If provided, searches for a valid subset of the target dataset where the mask is True.
            If not provided, a mask is automatically inferred from the nodata value of the target dataset.
            Defaults to None.
        reference_mask (xr.DataArray | None): A mask for the reference dataset.
            If provided, searches for a valid subset of the reference dataset where the mask is True.
            If not provided, a mask is automatically inferred from the nodata value of the reference dataset.
            Defaults to None.
        max_invalid_ratio (float): The maximum ratio of invalid pixels in the target and reference subset masks.
            Is not used if the masks are not provided.
            Defaults to 0.1.
        resample_to (Literal["reference", "target"] | None): The dataset to resample the other dataset to.
            If "reference", the target dataset is resampled to the reference dataset.
            If "target", the reference dataset is resampled to the target dataset.
            Defaults to None.
            If None, no resampling is done.
            This assumes that the pixel grids of the target and reference datasets are already aligned.

    Returns:
        list[tuple[plt.Figure, list[plt.Axes]] | None]: A list of tuples each containing a figure with its axes.
            Each figure shows a different step of the alignment process:
            1. The input target and reference datasets.
            2. In case of a CRS mismatch, the reprojected reference dataset before and after reprojection.
            3. The spatial intersection of the target and reference datasets.
            4. The target and reference datasets after resampling (if applicable).
            5. The target and reference subsets used for alignment.
            6. The cross power spectrum of the target and reference subsets.
            7. The final aligned target dataset.

    Raises:
        ValueError: If the target and reference are not both datasets or dataarrays.
        ValueError: If the target and reference datasets do not have dimensions 'x' and 'y'.

    """
    vizs = [None] * 6

    target = target.copy(deep=True)
    target_orig = target.copy(deep=True)

    # Check if both are datasets or dataarrays
    both_are_datasets = isinstance(target, xr.Dataset) and isinstance(reference, xr.Dataset)
    both_are_dataarrays = isinstance(target, xr.DataArray) and isinstance(reference, xr.DataArray)
    if not (both_are_datasets or both_are_dataarrays):
        raise ValueError("Both target and reference must be either xr.Dataset or xr.DataArray.")

    # Check if the dimentsions are x and y
    if "x" not in target.dims or "y" not in target.dims:
        raise ValueError("Target dataset must have dimensions 'x' and 'y'.")
    if "x" not in reference.dims or "y" not in reference.dims:
        raise ValueError("Reference dataset must have dimensions 'x' and 'y'.")

    # Get size-multipliers to reduce the size of the images for visualization
    # E.g. 4000x5000 -> 800x1000
    tm = max(1, target.sizes["x"] // 1000, target.sizes["y"] // 1000)
    rm = max(1, reference.sizes["x"] // 1000, reference.sizes["y"] // 1000)

    if both_are_datasets:
        # Only work on subset of the bands
        bands = _get_bands(target, reference, bands)
        # Apply bands to the datasets
        target = target[bands]
        reference = reference[bands]

        fig, axs = plt.subplots(2, len(bands), figsize=(len(bands) * 4, 6))
        for i, band in enumerate(bands):
            cmap = f"{band}s".capitalize() if band in ["red", "green", "blue"] else "gray"
            target[band][::tm, ::tm].plot(ax=axs[0, i], cmap=cmap)
            axs[0, i].set_title(f"Target - {band}")
            reference[band][::rm, ::rm].plot(ax=axs[1, i], cmap=cmap)
            axs[1, i].set_title(f"Reference - {band}")
        vizs[0] = (fig, axs)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        target[::tm, ::tm].plot(ax=axs[0], cmap="gray")
        axs[0].set_title("Target")
        reference[::rm, ::rm].plot(ax=axs[1], cmap="gray")
        axs[1].set_title("Reference")
        vizs[0] = (fig, axs)

    # Check for matching crs
    if target.odc.geobox.crs != reference.odc.geobox.crs:
        logger.warning(
            f"Target and reference datasets have different CRS: {target.odc.geobox.crs} vs {reference.odc.geobox.crs}. "
            "Reprojecting reference to target CRS."
        )
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        if both_are_dataarrays:
            reference[::rm, ::rm].plot(ax=axs[0], cmap="gray")
            axs[0].set_title("Reference before reprojection")
        else:
            reference[bands[0]][::rm, ::rm].plot(ax=axs[0], cmap="gray")
            axs[0].set_title(f"Reference {bands[0]} before reprojection")
        reference = reference.odc.reproject(target.odc.geobox.crs, resampling="nearest")
        reference_mask = (
            reference_mask.astype("int8").odc.reproject(target.odc.geobox.crs, resampling="nearest").astype("bool")
            if reference_mask is not None
            else None
        )
        if both_are_dataarrays:
            reference[::rm, ::rm].plot(ax=axs[1], cmap="gray")
            axs[1].set_title("Reference after reprojection")
        else:
            reference[bands[0]][::rm, ::rm].plot(ax=axs[1], cmap="gray")
            axs[1].set_title(f"Reference {bands[0]} after reprojection")
        vizs[1] = (fig, axs)

    # Get spatial intersection of the two images
    common_extent = reference.odc.geobox.geographic_extent.intersection(target.odc.geobox.geographic_extent)
    reference = reference.odc.crop(common_extent)
    target = target.odc.crop(common_extent)
    reference_mask = (
        reference_mask.odc.crop(common_extent).fillna(0.0).astype("bool") if reference_mask is not None else None
    )
    target_mask = target_mask.odc.crop(common_extent).fillna(0.0).astype("bool") if target_mask is not None else None

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    if both_are_dataarrays:
        target[::tm, ::tm].plot(ax=axs[0, 0], cmap="gray")
        axs[0, 0].set_title("Target - Spatial Intersection")
        reference[::rm, ::rm].plot(ax=axs[0, 1], cmap="gray")
        axs[0, 1].set_title("Reference - Spatial Intersection")
    else:
        target[bands[0]][::tm, ::tm].plot(ax=axs[0, 0], cmap="gray")
        axs[0, 0].set_title(f"Target {bands[0]} - Spatial Intersection")
        reference[bands[0]][::rm, ::rm].plot(ax=axs[0, 1], cmap="gray")
        axs[0, 1].set_title(f"Reference {bands[0]} - Spatial Intersection")
    if target_mask is not None:
        target_mask[::tm, ::tm].plot(ax=axs[1, 0], vmin=0, vmax=1)
        axs[1, 0].set_title("Target Mask - Spatial Intersection")
    if reference_mask is not None:
        reference_mask[::rm, ::rm].plot(ax=axs[1, 1], vmin=0, vmax=1)
        axs[1, 1].set_title("Reference Mask - Spatial Intersection")
    vizs[2] = (fig, axs)

    # Resample if requested
    # We can savely use nearest resampling here, since the datatype will be converted to complex64 anyway
    # And the matching algorithm works in the frequency domain, so using we use the fastest resampling method
    if resample_to == "reference":
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        if both_are_dataarrays:
            target[::tm, ::tm].plot(ax=axs[0], cmap="gray")
            axs[0].set_title("Target - Resampled to Reference")
        else:
            target[bands[0]][::tm, ::tm].plot(ax=axs[0], cmap="gray")
            axs[0].set_title(f"Target {bands[0]} - Resampled to Reference")
        target = target.odc.reproject(reference.odc.geobox, resampling="nearest")
        target_mask = (
            target_mask.astype("int8").odc.reproject(reference.odc.geobox, resampling="nearest").astype("bool")
            if target_mask is not None
            else None
        )
        if both_are_dataarrays:
            target[::tm, ::tm].plot(ax=axs[1], cmap="gray")
            axs[1].set_title("Target - Resampled to Reference")
        else:
            target[bands[0]][::tm, ::tm].plot(ax=axs[1], cmap="gray")
            axs[1].set_title(f"Target {bands[0]} - Resampled to Reference")
        vizs[3] = (fig, axs)
    elif resample_to == "target":
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        if both_are_dataarrays:
            reference[::rm, ::rm].plot(ax=axs[0], cmap="gray")
            axs[0].set_title("Reference - Resampled to Target")
        else:
            reference[bands[0]][::rm, ::rm].plot(ax=axs[0], cmap="gray")
            axs[0].set_title(f"Reference {bands[0]} - Resampled to Target")
        reference = reference.odc.reproject(target.odc.geobox, resampling="nearest")
        reference_mask = (
            reference_mask.astype("int8").odc.reproject(target.odc.geobox, resampling="nearest").astype("bool")
            if reference_mask is not None
            else None
        )
        if both_are_dataarrays:
            reference[::rm, ::rm].plot(ax=axs[1], cmap="gray")
            axs[1].set_title("Reference - Resampled to Target")
        else:
            reference[bands[0]][::rm, ::rm].plot(ax=axs[1], cmap="gray")
            axs[1].set_title(f"Reference {bands[0]} - Resampled to Target")
        vizs[3] = (fig, axs)

    if subset is None:
        target, reference = _find_suitable_subset(
            target,
            reference,
            window_size=window_size,
            target_mask=target_mask,
            reference_mask=reference_mask,
            max_invalid_ratio=max_invalid_ratio,
        )
        if reference is None or target is None:
            raise ValueError(
                "No suitable subset found for alignment. Try decreasing the window size and check the masks."
            )
    elif isinstance(subset, dict):
        reference = reference.isel(x=subset["x"], y=subset["y"])
        target = target.isel(x=subset["x"], y=subset["y"])

    if subset is None or isinstance(subset, dict):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        if both_are_dataarrays:
            target.plot(ax=axs[0, 0], cmap="gray")
            axs[0, 0].set_title("Target - Subset for Alignment")
            reference.plot(ax=axs[0, 1], cmap="gray")
            axs[0, 1].set_title("Reference - Subset for Alignment")
            target_mask.loc[{"x": target.x, "y": target.y}]
        else:
            target[bands[0]].plot(ax=axs[0, 0], cmap="gray")
            axs[0, 0].set_title(f"Target {bands[0]} - Subset for Alignment")
            reference[bands[0]].plot(ax=axs[0, 1], cmap="gray")
            axs[0, 1].set_title(f"Reference {bands[0]} - Subset for Alignment")
        if target_mask is not None:
            target_mask.sel(x=target.x, y=target.y).plot(ax=axs[1, 0], vmin=0, vmax=1)
            axs[1, 0].set_title("Target Mask - Subset for Alignment")
        if reference_mask is not None:
            reference_mask.sel(x=reference.x, y=reference.y).plot(ax=axs[1, 1], vmin=0, vmax=1)
            axs[1, 1].set_title("Reference Mask - Subset for Alignment")
        vizs[4] = (fig, axs)

        # Calculate the offset between the two subsets
    if both_are_dataarrays:
        shifted_cross_power_spectrum = _calculate_scps(reference, target)
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.imshow(shifted_cross_power_spectrum, cmap="inferno")
        axs.set_title("Cross Power Spectrum")
        vizs[5] = (fig, axs)
        # Find the peak in the cross power spectrum
        # The peak position in relation to the images center corresponds to the offset between the two images
        y_peak, x_peak = np.unravel_index(np.argmax(shifted_cross_power_spectrum), shifted_cross_power_spectrum.shape)
        x_offset = x_peak - reference.sizes["x"] // 2
        y_offset = y_peak - reference.sizes["y"] // 2
        print(f"Offset: x_offset={x_offset}, y_offset={y_offset}")
    else:
        offsets = {
            "x": [],
            "y": [],
        }
        fig, axs = plt.subplots(1, len(bands), figsize=(len(bands) * 4, 4))
        for i, band in enumerate(bands):
            shifted_cross_power_spectrum = _calculate_scps(reference[band], target[band])
            axs[i].imshow(shifted_cross_power_spectrum, cmap="inferno")
            axs[i].set_title(f"Cross Power Spectrum - {band}")
            # Find the peak in the cross power spectrum
            # The peak position in relation to the images center corresponds to the offset between the two images
            y_peak, x_peak = np.unravel_index(
                np.argmax(shifted_cross_power_spectrum), shifted_cross_power_spectrum.shape
            )
            x_offset = x_peak - reference.sizes["x"] // 2
            y_offset = y_peak - reference.sizes["y"] // 2
            offsets["x"].append(x_offset)
            offsets["y"].append(y_offset)
        vizs[5] = (fig, axs)

        # Calculate the average offset
        x_offset = np.mean(offsets["x"])
        y_offset = np.mean(offsets["y"])

        dbg_msg = f"Average offset: x_offset={x_offset}, y_offset={y_offset}"
        for i in range(len(bands)):
            dbg_msg += f"\n- Band {bands[i]}: x_offset={offsets['x'][i]}, y_offset={offsets['y'][i]}"
        print(dbg_msg)

    target_orig["x"] = target_orig.x + x_offset * target_orig.odc.geobox.resolution.x
    target_orig["y"] = target_orig.y + y_offset * target_orig.odc.geobox.resolution.y
    target_orig = target_orig.sel(x=reference.x, y=reference.y, method="nearest")
    if both_are_dataarrays:
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
        target.plot(ax=axs[0], cmap="gray")
        axs[0].set_title("Target")
        target_orig.plot(ax=axs[1], cmap="gray")
        axs[1].set_title("Target (aligned)")
        reference.plot(ax=axs[2], cmap="gray")
        axs[2].set_title("Reference")
        vizs.append((fig, axs))
    else:
        fig, axs = plt.subplots(len(bands), 3, figsize=(12, len(bands) * 4))
        for i, band in enumerate(bands):
            cmap = f"{band}s".capitalize() if band in ["red", "green", "blue"] else "gray"
            target[band].plot(ax=axs[i, 0], cmap=cmap)
            axs[i, 0].set_title(f"Target {band}")
            target_orig[band].plot(ax=axs[i, 1], cmap=cmap)
            axs[i, 1].set_title(f"Target {band} (aligned)")
            reference[band].plot(ax=axs[i, 2], cmap=cmap)
            axs[i, 2].set_title(f"Reference {band}")
        vizs.append((fig, axs))

    return vizs
