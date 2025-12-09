"""Utility functions for exporting debug data."""

from pathlib import Path

import rasterio as rio
import rioxarray
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
    output_path.mkdir(parents=True, exist_ok=True)
    optical = dataset[optical_bands].to_dataarray(dim="band").fillna(0).astype("uint16")
    optical.rio.to_raster(output_path / "optical_raw.tiff")

    with rio.open(output_path / "optical_raw.tiff", "r+") as rds:
        rds.descriptions = optical_bands

    band_info = "Optical Bands:\n"
    band_info += "\n".join([f" - {i + 1}: {band}" for i, band in enumerate(optical_bands)])

    if mask_bands:
        masks = dataset[mask_bands].to_dataarray(dim="band").fillna(0).astype("uint8")
        masks.rio.to_raster(output_path / "mask_raw.tiff")

        with rio.open(output_path / "mask_raw.tiff", "r+") as rds:
            rds.descriptions = mask_bands

        band_info += "\n\nMask Bands:\n"
        band_info += "\n".join([f" - {i + 1}: {band}" for i, band in enumerate(mask_bands)])

    (output_path / "bands.txt").write_text(band_info)
