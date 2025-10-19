"""Utility functions for exporting debug data."""

from pathlib import Path

import xarray as xr


def save_debug_geotiff(
    dataset: xr.Dataset,
    output_path: Path,
    optical_bands: list[str],
    mask_bands: list[str] | None = None,
) -> None:
    """Save the raw dataset as a GeoTIFF file for debugging purposes.

    Args:
        dataset (xr.Dataset): Dataset to save
        output_path (Path): Path to the output GeoTIFF file
        optical_bands (list[str]): List of optical band names
        mask_bands (list[str]): List of mask band names

    """
    optical = dataset[optical_bands].to_dataarray(dim="band").fillna(0).astype("uint16")
    optical.rio.to_raster(output_path / "optical_raw.tiff")
    if mask_bands:
        masks = dataset[mask_bands].to_dataarray(dim="band").fillna(0).astype("uint8")
        masks.rio.to_raster(output_path / "mask_raw.tiff")
