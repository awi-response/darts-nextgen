"""Re-implementation of the AROSICS algorithm."""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
from scipy.fft import fft2, fftshift, ifft2
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__.replace("darts_", "darts."))


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


def _find_suitable_subset_slices(
    target: xr.Dataset | xr.DataArray,
    reference: xr.Dataset | xr.DataArray,
    window_size: int = 256,
    target_mask: xr.DataArray | None = None,
    reference_mask: xr.DataArray | None = None,
    max_invalid_ratio: float = 0.01,
) -> dict[Literal["x", "y"], slice] | Literal[False]:
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

    # This will search for a valid subset in a spiraling pattern
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
            return {"x": xslice, "y": yslice}
        # If not valid, shift the corner in a spiraling pattern
        corner = (corner[0] + dx[direction], corner[1] + dy[direction])

        # Check if we are still in bounds
        if (corner[0] < 0 or corner[0] + window_size > target.sizes["x"]) or (
            corner[1] < 0 or corner[1] + window_size > target.sizes["y"]
        ):
            logger.debug(
                "Couldn't find a valid subset in the target and reference datasets."
                " Please check the window size and masks. Falling back to calculate offsets with the complete images."
            )
            break

        # Update direction if needed
        steps_in_direction += 1
        if steps_in_direction == turns_taken // 2 + 1:
            direction = (direction + 1) % 4
            steps_in_direction = 0
            turns_taken += 1
    return False


def _get_window_subsets(
    target: xr.DataArray,
    reference: xr.DataArray,
    subset: dict[Literal["x", "y"], slice] | Literal[False],
) -> tuple[xr.DataArray, xr.DataArray]:
    if not subset:
        # In case of no subset, we need to find the common spatial intersection of the two images again, since the
        # Target could be changed since inbetween calls
        common_extent = reference.odc.geobox.geographic_extent.intersection(target.odc.geobox.geographic_extent)
        reference_window = reference.odc.crop(common_extent)
    else:
        reference_window = reference.isel(**subset)
    target_window = target.sel(x=reference_window.x, y=reference_window.y, method="nearest")
    return target_window, reference_window


def _calculate_scps(reference_window: xr.DataArray, target_window: xr.DataArray) -> xr.DataArray:
    # Calculate the shifted cross power spectrum
    # This is a trick to avoid convoluted block matching:
    # Convolutions can be computed in the frequency domain with fourier transforms
    # Hence, we turn our images into the frequency domain, compute the cross power spectrum,
    # and then turn it back into the spatial domain.
    # The peak of the result tells us the spatial shift between the two images.
    ref_freq = fft2(reference_window.fillna(0).astype("complex64"))
    target_freq = fft2(target_window.fillna(0).astype("complex64"))
    eps = np.abs(ref_freq).max() * 1e-15
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_power_spectrum = (ref_freq * target_freq.conj()) / (abs(ref_freq) * abs(target_freq) + eps)
    cross_power_spectrum = ifft2(cross_power_spectrum)
    cross_power_spectrum = abs(cross_power_spectrum)
    shifted_cross_power_spectrum = fftshift(cross_power_spectrum)
    shifted_cross_power_spectrum = reference_window.copy(data=shifted_cross_power_spectrum)
    return shifted_cross_power_spectrum


@dataclass
class OffsetInfo:
    """Dataclass to hold offset information."""

    x_offset: float | None
    y_offset: float | None
    ssim_before: float | None = None
    ssim_after: float | None = None
    shift_reliability: float = 0.0

    @property
    def ssim_improved(self) -> bool:  # noqa: D102
        return self.ssim_before <= self.ssim_after

    @property
    def ssim_improvement(self) -> float:  # noqa: D102
        return self.ssim_after - self.ssim_before

    @property
    def xy(self, max_offset: float = 10.0, min_reliability: float = 30.0) -> tuple[float, float]:  # noqa: D102
        if not self.is_valid(max_offset=max_offset, min_reliability=min_reliability):
            return 0.0, 0.0
        return self.x_offset or 0.0, self.y_offset or 0.0

    def is_valid(self, max_offset: float, min_reliability: float) -> bool:  # noqa: D102
        if self.x_offset is None or self.y_offset is None:
            return False
        return (
            abs(self.x_offset) <= max_offset
            and abs(self.y_offset) <= max_offset
            and self.shift_reliability >= min_reliability
            and self.ssim_after >= self.ssim_before
        )


@dataclass
class MultiOffsetInfo:
    """Dataclass to hold offset information of multiple bands."""

    x_offset: float | None
    y_offset: float | None
    offset_infos: dict[str, OffsetInfo]

    def is_valid(self):  # noqa: D102
        return self.x_offset is not None and self.y_offset is not None

    def to_dataframe(self):  # noqa: D102
        import pandas as pd

        df = pd.DataFrame.from_records(
            [
                {
                    "band": band,
                    "x_offset": oi.x_offset,
                    "y_offset": oi.y_offset,
                    "ssim_before": oi.ssim_before,
                    "ssim_after": oi.ssim_after,
                    "shift_reliability": oi.shift_reliability,
                }
                for band, oi in self.offset_infos.items()
            ]
        )
        return df

    @classmethod
    def _from_offsets(
        cls,
        offset_infos: dict[str, OffsetInfo],
        max_offset: float,
        min_reliability: float,
    ) -> "MultiOffsetInfo":
        # Try to resolve the best offset based on a multiple offsets
        # 0. Filter invalids
        # 1. Check if all offsets are (almost) equal (within a single pixel) -> take mean
        # 2. Check if all offsets are somewhat close (low std) -> take weighted mean based on reliability
        # 3. Use best offset based on reliability
        for band, oi in offset_infos.items():
            if not oi.is_valid(max_offset=max_offset, min_reliability=min_reliability):
                logger.debug(f"{band=} resulted in an invalid offset: {oi}")

        x_offsets = np.array(
            [
                oi.x_offset
                for oi in offset_infos.values()
                if oi.is_valid(max_offset=max_offset, min_reliability=min_reliability)
            ]
        )
        y_offsets = np.array(
            [
                oi.y_offset
                for oi in offset_infos.values()
                if oi.is_valid(max_offset=max_offset, min_reliability=min_reliability)
            ]
        )
        reliabilities = np.array(
            [
                oi.shift_reliability
                for oi in offset_infos.values()
                if oi.is_valid(max_offset=max_offset, min_reliability=min_reliability)
            ]
        )
        if len(x_offsets) == 0 or len(y_offsets) == 0:
            logger.warning("No valid offsets found. Returning 0.")
            return MultiOffsetInfo(x_offset=None, y_offset=None, offset_infos=offset_infos)

        x_max_deviation = abs(x_offsets - x_offsets.mean()).max()
        y_max_deviation = abs(y_offsets - y_offsets.mean()).max()
        if x_max_deviation < 1 and y_max_deviation < 1:
            return MultiOffsetInfo(x_offset=x_offsets.mean(), y_offset=y_offsets.mean(), offset_infos=offset_infos)

        if np.std(x_offsets) < 1 and np.std(y_offsets) < 1:
            return MultiOffsetInfo(
                x_offset=np.average(x_offsets, weights=reliabilities),
                y_offset=np.average(y_offsets, weights=reliabilities),
                offset_infos=offset_infos,
            )

        # Use the best offset based on reliability
        best = reliabilities.argmax()
        return MultiOffsetInfo(x_offset=x_offsets[best], y_offset=y_offsets[best], offset_infos=offset_infos)


def _calculate_offset(
    reference: xr.DataArray,
    target: xr.DataArray,
    subset: dict[Literal["x", "y"], slice] | Literal[False] = False,
    max_iter: int = 5,
) -> OffsetInfo:
    # We need to transpose the arrays to ensure that it is always (y, x) since we operate outside of xarray
    # This also ensures that altering the axes here does not change the original arrays axes
    target = target.transpose("y", "x")
    reference = reference.transpose("y", "x")

    # Calculate the ssim before the offset calculation
    ssim_before = _calc_ssim(target, reference, subset=subset)

    potential_x_offset = 0
    potential_y_offset = 0
    for i in range(max_iter):
        target_window, reference_window = _get_window_subsets(target, reference, subset)
        shifted_cross_power_spectrum = _calculate_scps(reference_window, target_window)
        # Find the peak in the cross power spectrum
        # The peak position in relation to the images center corresponds to the offset between the two images
        # Since we use unravel_index, it is important that the scps is in (y, x) order -> transposed in the beginning
        y_peak, x_peak = np.unravel_index(shifted_cross_power_spectrum.argmax(), shifted_cross_power_spectrum.shape)
        x_offset = x_peak - reference_window.sizes["x"] // 2
        y_offset = y_peak - reference_window.sizes["y"] // 2
        # If no more offset is left, it means that we found the "real" offset in the iteration before
        # in this case we break and continue with subpixel offset calculation and validation
        if x_offset == 0 and y_offset == 0:
            break

        # Apply the offset to the target array
        target["x"] = target.x + x_offset * target.odc.geobox.resolution.x
        target["y"] = target.y + y_offset * target.odc.geobox.resolution.y

        # Add to the overall offset
        potential_x_offset += x_offset
        potential_y_offset += y_offset
    else:
        logger.debug(f"Could not find a suitable offset after {max_iter} iterations.")
        return OffsetInfo(x_offset=None, y_offset=None)

    # Calculate subpixel shift
    sm_left = shifted_cross_power_spectrum.isel(x=x_peak - 1, y=y_peak).item()
    sm_right = shifted_cross_power_spectrum.isel(x=x_peak + 1, y=y_peak).item()
    sm_above = shifted_cross_power_spectrum.isel(x=x_peak, y=y_peak - 1).item()
    sm_below = shifted_cross_power_spectrum.isel(x=x_peak, y=y_peak + 1).item()

    v_00 = shifted_cross_power_spectrum.max().item()
    v_10 = max(sm_left, sm_right)  # x
    v_01 = max(sm_above, sm_below)  # y

    subpixel_x_offset = np.sign(sm_right - sm_left) * v_10 / (v_00 + v_10)
    subpixel_y_offset = np.sign(sm_below - sm_above) * v_01 / (v_00 + v_01)
    subpixel_x_offset = np.round(subpixel_x_offset, 3)
    subpixel_y_offset = np.round(subpixel_y_offset, 3)

    # Apply the offset to the target array
    target["x"] = target.x + subpixel_x_offset * target.odc.geobox.resolution.x
    target["y"] = target.y + subpixel_y_offset * target.odc.geobox.resolution.y

    potential_x_offset += subpixel_x_offset
    potential_y_offset += subpixel_y_offset

    # Calculate metrics
    shift_reliability = _calc_shift_reliability(shifted_cross_power_spectrum, x_peak, y_peak)

    # Calculate the ssim after the offsets are applied
    ssim_after = _calc_ssim(target=target, reference=reference, subset=subset)

    return OffsetInfo(
        x_offset=potential_x_offset,
        y_offset=potential_y_offset,
        shift_reliability=shift_reliability,
        ssim_before=ssim_before,
        ssim_after=ssim_after,
    )


def _calc_shift_reliability(shifted_cross_power_spectrum: xr.DataArray, x_peak: int, y_peak: int) -> float:
    # calculate mean power at peak
    x_peak_slice = slice(x_peak - 1, x_peak + 2)
    y_peak_slice = slice(y_peak - 1, y_peak + 2)
    power_at_peak = shifted_cross_power_spectrum.isel(x=x_peak_slice, y=y_peak_slice).mean().item()

    # calculate mean power without peak + 3* standard deviation
    shifted_cross_power_spectrum_unpeaked = shifted_cross_power_spectrum.copy()
    shifted_cross_power_spectrum_unpeaked[x_peak_slice, y_peak_slice] = -9999
    shifted_cross_power_spectrum_unpeaked = np.ma.masked_equal(shifted_cross_power_spectrum_unpeaked.values, -9999)
    power_without_peak = np.mean(shifted_cross_power_spectrum_unpeaked) + 2 * np.std(
        shifted_cross_power_spectrum_unpeaked
    )

    # calculate confidence
    shift_reliability = 100 - ((power_without_peak / power_at_peak) * 100)
    shift_reliability = min(max(shift_reliability, 0), 100)

    return shift_reliability


def _calc_ssim(
    target: xr.DataArray, reference: xr.DataArray, subset: dict[Literal["x", "y"], slice] | Literal[False]
) -> float:
    target_window, reference_window = _get_window_subsets(target, reference, subset)

    # Normalise both arrays
    target_window = (target_window - target_window.min()) / (target_window.max() - target_window.min())
    reference_window = (reference_window - reference_window.min()) / (reference_window.max() - reference_window.min())
    # Mask NaN values
    target_window = np.ma.masked_array(target_window.astype(np.float64).values, mask=target_window.isnull())
    reference_window = np.ma.masked_array(reference_window.astype(np.float64).values, mask=reference_window.isnull())
    return ssim(target_window, reference_window, data_range=1)


def get_offsets(
    target: xr.Dataset | xr.DataArray,
    reference: xr.Dataset | xr.DataArray,
    bands: list[str] | Literal["multiband"] | str = "multiband",
    subset: dict[Literal["x", "y"], slice] | Literal[False] | None = None,
    window_size: int = 256,
    target_mask: xr.DataArray | None = None,
    reference_mask: xr.DataArray | None = None,
    max_invalid_ratio: float = 0.01,
    max_iter: int = 5,
    min_reliability: float = 30.0,
    max_offset: float = 10.0,
    resample_to: Literal["reference", "target"] | None = None,
) -> OffsetInfo | MultiOffsetInfo:
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
        max_iter (int): The maximum number of iterations to find the offset. Defaults to 5.
        min_reliability (float): The minimum reliability (in %) of the offset to consider it valid.
            Defaults to 30.0.
        max_offset (float): The maximum offset in pixels to consider the offset valid.
            Defaults to 10.0.
        resample_to (Literal["reference", "target"] | None): The dataset to resample the other dataset to.
            If "reference", the target dataset is resampled to the reference dataset.
            If "target", the reference dataset is resampled to the target dataset.
            Defaults to None.
            If None, no resampling is done.
            This assumes that the pixel grids of the target and reference datasets are already aligned.

    Returns:
        OffsetInfo | MultiOffsetInfo: The offsets in x and y direction between the target and reference datasets.

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
        subset = _find_suitable_subset_slices(
            target,
            reference,
            window_size=window_size,
            target_mask=target_mask,
            reference_mask=reference_mask,
            max_invalid_ratio=max_invalid_ratio,
        )

    # Calculate the offset between the two subsets
    if both_are_dataarrays:
        offset_info = _calculate_offset(reference, target, subset=subset, max_iter=max_iter)
        return offset_info
    else:
        offsets = {
            band: _calculate_offset(reference[band], target[band], subset=subset, max_iter=max_iter) for band in bands
        }
        return MultiOffsetInfo._from_offsets(offsets, max_offset=max_offset, min_reliability=min_reliability)


def align(
    target: xr.Dataset | xr.DataArray,
    reference: xr.Dataset | xr.DataArray,
    bands: list[str] | Literal["multiband"] | str = "multiband",
    subset: dict[Literal["x", "y"], slice] | Literal[False] | None = None,
    window_size: int = 256,
    target_mask: xr.DataArray | None = None,
    reference_mask: xr.DataArray | None = None,
    max_invalid_ratio: float = 0.01,
    max_iter: int = 5,
    min_reliability: float = 30.0,
    max_offset: float = 10.0,
    resample_to: Literal["reference", "target"] | None = None,
    return_offset: bool = False,
    round_axes: int | Literal[False] = 3,
    inplace: bool = False,
) -> tuple[xr.Dataset | xr.DataArray, OffsetInfo | MultiOffsetInfo] | xr.Dataset | xr.DataArray:
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
        max_iter (int): The maximum number of iterations to find the offset. Defaults to 5.
        min_reliability (float): The minimum reliability (in %) of the offset to consider it valid.
            Defaults to 30.0.
        max_offset (float): The maximum offset in pixels to consider the offset valid.
            Defaults to 10.0.
        resample_to (Literal["reference", "target"] | None): The dataset to resample the other dataset to.
            If "reference", the target dataset is resampled to the reference dataset.
            If "target", the reference dataset is resampled to the target dataset.
            Defaults to None.
            If None, no resampling is done.
            This assumes that the pixel grids of the target and reference datasets are already aligned.
        round_axes (int | False): The number of decimal places to round the x and y coordinates of the target dataset.
            This may be necessary if the applying of offsets results in very small floating point errors
            that lead to misalignment when using further processing steps.
            If False, no rounding is done.
            Defaults to 3.
        return_offset (bool): If True, returns the offsets instead of aligning the target dataset.
        inplace (bool): If True, modifies the target dataset in place.

    Returns:
        tuple[xr.Dataset | xr.DataArray, OffsetInfo | MultiOffsetInfo] | xr.Dataset | xr.DataArray:
            The aligned target dataset or dataarray.
            If return_offset is True, also returns the offsets in x and y direction as a tuple.

    """
    offset_info = get_offsets(
        target,
        reference,
        bands=bands,
        subset=subset,
        window_size=window_size,
        target_mask=target_mask,
        reference_mask=reference_mask,
        max_invalid_ratio=max_invalid_ratio,
        max_iter=max_iter,
        max_offset=max_offset,
        min_reliability=min_reliability,
        resample_to=resample_to,
    )

    invalid_single = isinstance(offset_info, OffsetInfo) and not offset_info.is_valid(
        max_offset=max_offset, min_reliability=min_reliability
    )
    invalid_multi = isinstance(offset_info, MultiOffsetInfo) and not offset_info.is_valid()
    if invalid_single or invalid_multi:
        logger.warning("No valid offset found. Returning the original target dataset.")
        return target, offset_info

    x_offset = offset_info.x_offset
    y_offset = offset_info.y_offset

    if (x_offset == 0 and y_offset == 0) or (x_offset is None or y_offset is None):
        return target, offset_info

    # Apply the offset to the target dataset
    if not inplace:
        target = target.copy(deep=True)
    target["x"] = target.x + x_offset * target.odc.geobox.resolution.x
    target["y"] = target.y + y_offset * target.odc.geobox.resolution.y

    if round_axes is not False:
        target["x"] = target.x.round(round_axes)
        target["y"] = target.y.round(round_axes)

    if return_offset:
        return target, offset_info
    else:
        return target


# The visualization are note up to date
# Would be best to rewrite them / split up into multiple parts:
# e.g. one for the setup part (finding intersection, resampling, etc.)
# one for the alignment part
# and one final one for the result

# def visualize_alignment(
#     target: xr.Dataset | xr.DataArray,
#     reference: xr.Dataset | xr.DataArray,
#     bands: list[str] | Literal["multiband"] | str = "multiband",
#     subset: dict[Literal["x", "y"], slice] | Literal[False] | None = None,
#     window_size: int = 256,
#     target_mask: xr.DataArray | None = None,
#     reference_mask: xr.DataArray | None = None,
#     max_invalid_ratio: float = 0.01,
#     resample_to: Literal["reference", "target"] | None = None,
# ) -> list[tuple[plt.Figure, list[plt.Axes]] | None]:
#     """Visualized the alignment inner workings.

#     Note:
#         This function never works inplace!

#     Args:
#         target (xr.Dataset | xr.DataArray): The target image dataset or dataarray to be aligned.
#         reference (xr.Dataset | xr.DataArray): The reference image dataset or dataarray.
#         bands (list[str] | Literal["multiband"] | str): The bands to use for alignment.
#             Only used if the target and reference are datasets.
#             If "multiband", all bands are used.
#             This expects the target and reference datasets to have the same band names.
#             If string, the respective band is used for alignment.
#             If a list of strings, only the specified bands are used for alignment.
#             Note: All bands are shifted by the same offset, even when using "multiband".
#             With multiband, the offset is calculated from the average of all common and valid band offsets.
#             This is slower but more robust than using a single band.
#             If a band-specific offset is desired,
#             please use the `get_dataarray_offsets` function for each band separately.
#             Defaults to "multiband".
#         subset (dict[Literal["x", "y"], slice] | Literal[False] | None): A dictionary of slices to use for alignment.
#             If provided, only the specified subset of the target and reference datasets is used for alignment.
#             The dictionary must contain the keys "x" and "y" with the respective slices.
#             If False, the whole dataset is used for alignment.
#             If None, will try to find a suitable subset automatically.
#         window_size (int): The size of the window to use for alignment in case no subset is provided. Defaults to 256.
#         target_mask (xr.DataArray | None): A mask for the target dataset.
#             If provided, searches for a valid subset of the target dataset where the mask is True.
#             If not provided, a mask is automatically inferred from the nodata value of the target dataset.
#             Defaults to None.
#         reference_mask (xr.DataArray | None): A mask for the reference dataset.
#             If provided, searches for a valid subset of the reference dataset where the mask is True.
#             If not provided, a mask is automatically inferred from the nodata value of the reference dataset.
#             Defaults to None.
#         max_invalid_ratio (float): The maximum ratio of invalid pixels in the target and reference subset masks.
#             Is not used if the masks are not provided.
#             Defaults to 0.1.
#         resample_to (Literal["reference", "target"] | None): The dataset to resample the other dataset to.
#             If "reference", the target dataset is resampled to the reference dataset.
#             If "target", the reference dataset is resampled to the target dataset.
#             Defaults to None.
#             If None, no resampling is done.
#             This assumes that the pixel grids of the target and reference datasets are already aligned.

#     Returns:
#         list[tuple[plt.Figure, list[plt.Axes]] | None]: A list of tuples each containing a figure with its axes.
#             Each figure shows a different step of the alignment process:
#             1. The input target and reference datasets.
#             2. In case of a CRS mismatch, the reprojected reference dataset before and after reprojection.
#             3. The spatial intersection of the target and reference datasets.
#             4. The target and reference datasets after resampling (if applicable).
#             5. The target and reference subsets used for alignment.
#             6. The cross power spectrum of the target and reference subsets.
#             7. The final aligned target dataset.

#     Raises:
#         ValueError: If the target and reference are not both datasets or dataarrays.
#         ValueError: If the target and reference datasets do not have dimensions 'x' and 'y'.

#     """
#     vizs = [None] * 6

#     target = target.copy(deep=True)
#     target_orig = target.copy(deep=True)

#     # Check if both are datasets or dataarrays
#     both_are_datasets = isinstance(target, xr.Dataset) and isinstance(reference, xr.Dataset)
#     both_are_dataarrays = isinstance(target, xr.DataArray) and isinstance(reference, xr.DataArray)
#     if not (both_are_datasets or both_are_dataarrays):
#         raise ValueError("Both target and reference must be either xr.Dataset or xr.DataArray.")

#     # Check if the dimentsions are x and y
#     if "x" not in target.dims or "y" not in target.dims:
#         raise ValueError("Target dataset must have dimensions 'x' and 'y'.")
#     if "x" not in reference.dims or "y" not in reference.dims:
#         raise ValueError("Reference dataset must have dimensions 'x' and 'y'.")

#     # Get size-multipliers to reduce the size of the images for visualization
#     # E.g. 4000x5000 -> 800x1000
#     tm = max(1, target.sizes["x"] // 1000, target.sizes["y"] // 1000)
#     rm = max(1, reference.sizes["x"] // 1000, reference.sizes["y"] // 1000)

#     if both_are_datasets:
#         # Only work on subset of the bands
#         bands = _get_bands(target, reference, bands)
#         # Apply bands to the datasets
#         target = target[bands]
#         reference = reference[bands]

#         fig, axs = plt.subplots(2, len(bands), figsize=(len(bands) * 4, 6))
#         for i, band in enumerate(bands):
#             cmap = f"{band}s".capitalize() if band in ["red", "green", "blue"] else "gray"
#             target[band][::tm, ::tm].plot(ax=axs[0, i], cmap=cmap)
#             axs[0, i].set_title(f"Target - {band}")
#             reference[band][::rm, ::rm].plot(ax=axs[1, i], cmap=cmap)
#             axs[1, i].set_title(f"Reference - {band}")
#         vizs[0] = (fig, axs)
#     else:
#         fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#         target[::tm, ::tm].plot(ax=axs[0], cmap="gray")
#         axs[0].set_title("Target")
#         reference[::rm, ::rm].plot(ax=axs[1], cmap="gray")
#         axs[1].set_title("Reference")
#         vizs[0] = (fig, axs)

#     # Check for matching crs
#     if target.odc.geobox.crs != reference.odc.geobox.crs:
#         fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#         if both_are_dataarrays:
#             reference[::rm, ::rm].plot(ax=axs[0], cmap="gray")
#             axs[0].set_title("Reference before reprojection")
#         else:
#             reference[bands[0]][::rm, ::rm].plot(ax=axs[0], cmap="gray")
#             axs[0].set_title(f"Reference {bands[0]} before reprojection")
#         reference = reference.odc.reproject(target.odc.geobox.crs, resampling="nearest")
#         reference_mask = (
#             reference_mask.astype("int8").odc.reproject(target.odc.geobox.crs, resampling="nearest").astype("bool")
#             if reference_mask is not None
#             else None
#         )
#         if both_are_dataarrays:
#             reference[::rm, ::rm].plot(ax=axs[1], cmap="gray")
#             axs[1].set_title("Reference after reprojection")
#         else:
#             reference[bands[0]][::rm, ::rm].plot(ax=axs[1], cmap="gray")
#             axs[1].set_title(f"Reference {bands[0]} after reprojection")
#         vizs[1] = (fig, axs)

#     # Get spatial intersection of the two images
#     common_extent = reference.odc.geobox.geographic_extent.intersection(target.odc.geobox.geographic_extent)
#     reference = reference.odc.crop(common_extent)
#     target = target.odc.crop(common_extent)
#     reference_mask = (
#         reference_mask.odc.crop(common_extent).fillna(0.0).astype("bool") if reference_mask is not None else None
#     )
#     target_mask = target_mask.odc.crop(common_extent).fillna(0.0).astype("bool") if target_mask is not None else None

#     fig, axs = plt.subplots(2, 2, figsize=(8, 8))
#     if both_are_dataarrays:
#         target[::tm, ::tm].plot(ax=axs[0, 0], cmap="gray")
#         axs[0, 0].set_title("Target - Spatial Intersection")
#         reference[::rm, ::rm].plot(ax=axs[0, 1], cmap="gray")
#         axs[0, 1].set_title("Reference - Spatial Intersection")
#     else:
#         target[bands[0]][::tm, ::tm].plot(ax=axs[0, 0], cmap="gray")
#         axs[0, 0].set_title(f"Target {bands[0]} - Spatial Intersection")
#         reference[bands[0]][::rm, ::rm].plot(ax=axs[0, 1], cmap="gray")
#         axs[0, 1].set_title(f"Reference {bands[0]} - Spatial Intersection")
#     if target_mask is not None:
#         target_mask[::tm, ::tm].plot(ax=axs[1, 0], vmin=0, vmax=1)
#         axs[1, 0].set_title("Target Mask - Spatial Intersection")
#     if reference_mask is not None:
#         reference_mask[::rm, ::rm].plot(ax=axs[1, 1], vmin=0, vmax=1)
#         axs[1, 1].set_title("Reference Mask - Spatial Intersection")
#     vizs[2] = (fig, axs)

#     # Resample if requested
#     # We can savely use nearest resampling here, since the datatype will be converted to complex64 anyway
#     # And the matching algorithm works in the frequency domain, so using we use the fastest resampling method
#     if resample_to == "reference":
#         fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#         if both_are_dataarrays:
#             target[::tm, ::tm].plot(ax=axs[0], cmap="gray")
#             axs[0].set_title("Target - Resampled to Reference")
#         else:
#             target[bands[0]][::tm, ::tm].plot(ax=axs[0], cmap="gray")
#             axs[0].set_title(f"Target {bands[0]} - Resampled to Reference")
#         target = target.odc.reproject(reference.odc.geobox, resampling="nearest")
#         target_mask = (
#             target_mask.astype("int8").odc.reproject(reference.odc.geobox, resampling="nearest").astype("bool")
#             if target_mask is not None
#             else None
#         )
#         if both_are_dataarrays:
#             target[::tm, ::tm].plot(ax=axs[1], cmap="gray")
#             axs[1].set_title("Target - Resampled to Reference")
#         else:
#             target[bands[0]][::tm, ::tm].plot(ax=axs[1], cmap="gray")
#             axs[1].set_title(f"Target {bands[0]} - Resampled to Reference")
#         vizs[3] = (fig, axs)
#     elif resample_to == "target":
#         fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#         if both_are_dataarrays:
#             reference[::rm, ::rm].plot(ax=axs[0], cmap="gray")
#             axs[0].set_title("Reference - Resampled to Target")
#         else:
#             reference[bands[0]][::rm, ::rm].plot(ax=axs[0], cmap="gray")
#             axs[0].set_title(f"Reference {bands[0]} - Resampled to Target")
#         reference = reference.odc.reproject(target.odc.geobox, resampling="nearest")
#         reference_mask = (
#             reference_mask.astype("int8").odc.reproject(target.odc.geobox, resampling="nearest").astype("bool")
#             if reference_mask is not None
#             else None
#         )
#         if both_are_dataarrays:
#             reference[::rm, ::rm].plot(ax=axs[1], cmap="gray")
#             axs[1].set_title("Reference - Resampled to Target")
#         else:
#             reference[bands[0]][::rm, ::rm].plot(ax=axs[1], cmap="gray")
#             axs[1].set_title(f"Reference {bands[0]} - Resampled to Target")
#         vizs[3] = (fig, axs)

#     if subset is None:
#         subset = _find_suitable_subset_slices(
#             target,
#             reference,
#             window_size=window_size,
#             target_mask=target_mask,
#             reference_mask=reference_mask,
#             max_invalid_ratio=max_invalid_ratio,
#         )

#     if isinstance(subset, dict):
#         target_window, reference_window = _get_window_subsets(target, reference, subset)
#         fig, axs = plt.subplots(2, 2, figsize=(8, 8))
#         if both_are_dataarrays:
#             target_window.plot(ax=axs[0, 0], cmap="gray")
#             axs[0, 0].set_title("Target - Subset for Alignment")
#             reference_window.plot(ax=axs[0, 1], cmap="gray")
#             axs[0, 1].set_title("Reference - Subset for Alignment")
#         else:
#             target_window[bands[0]].plot(ax=axs[0, 0], cmap="gray")
#             axs[0, 0].set_title(f"Target {bands[0]} - Subset for Alignment")
#             reference_window[bands[0]].plot(ax=axs[0, 1], cmap="gray")
#             axs[0, 1].set_title(f"Reference {bands[0]} - Subset for Alignment")
#         if target_mask is not None:
#             target_mask.sel(x=target_window.x, y=target_window.y).plot(ax=axs[1, 0], vmin=0, vmax=1)
#             axs[1, 0].set_title("Target Mask - Subset for Alignment")
#         if reference_mask is not None:
#             reference_mask.sel(x=reference_window.x, y=reference_window.y).plot(ax=axs[1, 1], vmin=0, vmax=1)
#             axs[1, 1].set_title("Reference Mask - Subset for Alignment")
#         vizs[4] = (fig, axs)

#         # Calculate the offset between the two subsets
#     if both_are_dataarrays:
#         target_window, reference_window = _get_window_subsets(target, reference, subset)
#         shifted_cross_power_spectrum = _calculate_scps(reference_window, target_window)
#         fig, axs = plt.subplots(1, 1, figsize=(6, 6))
#         axs.imshow(shifted_cross_power_spectrum, cmap="inferno")
#         axs.set_title("Cross Power Spectrum")
#         vizs[5] = (fig, axs)
#         # Find the peak in the cross power spectrum
#         # The peak position in relation to the images center corresponds to the offset between the two images
#         y_peak, x_peak = np.unravel_index(np.argmax(shifted_cross_power_spectrum), shifted_cross_power_spectrum.shape)
#         x_offset = x_peak - reference.sizes["x"] // 2
#         y_offset = y_peak - reference.sizes["y"] // 2
#     else:
#         offsets = {
#             "x": [],
#             "y": [],
#         }
#         fig, axs = plt.subplots(1, len(bands), figsize=(len(bands) * 4, 4))
#         for i, band in enumerate(bands):
#             target_window, reference_window = _get_window_subsets(target[band], reference[band], subset)
#             shifted_cross_power_spectrum = _calculate_scps(reference_window, target_window)
#             axs[i].imshow(shifted_cross_power_spectrum, cmap="inferno")
#             axs[i].set_title(f"Cross Power Spectrum - {band}")
#             # Find the peak in the cross power spectrum
#             # The peak position in relation to the images center corresponds to the offset between the two images
#             y_peak, x_peak = np.unravel_index(
#               shifted_cross_power_spectrum.argmax(),
#               shifted_cross_power_spectrum.shape,
#             )
#             x_offset = x_peak - reference.sizes["x"] // 2
#             y_offset = y_peak - reference.sizes["y"] // 2
#             offsets["x"].append(x_offset)
#             offsets["y"].append(y_offset)
#         vizs[5] = (fig, axs)

#         # Calculate the average offset
#         x_offset = np.mean(offsets["x"])
#         y_offset = np.mean(offsets["y"])

#         dbg_msg = f"Average offset: x_offset={x_offset}, y_offset={y_offset}"
#         for i in range(len(bands)):
#             dbg_msg += f"\n- Band {bands[i]}: x_offset={offsets['x'][i]}, y_offset={offsets['y'][i]}"

#     target_orig["x"] = target_orig.x + x_offset * target_orig.odc.geobox.resolution.x
#     target_orig["y"] = target_orig.y + y_offset * target_orig.odc.geobox.resolution.y
#     target_orig = target_orig.sel(x=reference.x, y=reference.y, method="nearest")
#     if both_are_dataarrays:
#         fig, axs = plt.subplots(1, 3, figsize=(12, 6))
#         target.plot(ax=axs[0], cmap="gray")
#         axs[0].set_title("Target")
#         target_orig.plot(ax=axs[1], cmap="gray")
#         axs[1].set_title("Target (aligned)")
#         reference.plot(ax=axs[2], cmap="gray")
#         axs[2].set_title("Reference")
#         vizs.append((fig, axs))
#     else:
#         fig, axs = plt.subplots(len(bands), 3, figsize=(12, len(bands) * 4))
#         for i, band in enumerate(bands):
#             cmap = f"{band}s".capitalize() if band in ["red", "green", "blue"] else "gray"
#             target[band].plot(ax=axs[i, 0], cmap=cmap)
#             axs[i, 0].set_title(f"Target {band}")
#             target_orig[band].plot(ax=axs[i, 1], cmap=cmap)
#             axs[i, 1].set_title(f"Target {band} (aligned)")
#             reference[band].plot(ax=axs[i, 2], cmap=cmap)
#             axs[i, 2].set_title(f"Reference {band}")
#         vizs.append((fig, axs))

#     return vizs
