"""PLANET scene based preprocessing."""

from pathlib import Path

import xarray as xr

from darts_preprocessing.utils.data_pre_processing import (
    calculate_ndvi,
    load_auxiliary,
    load_data_masks,
    load_planet_scene,
)


def load_and_preprocess_planet_scene(planet_scene_path: Path, elevation_path: Path, slope_path: Path) -> xr.Dataset:
    """Load and preprocess a Planet Scene (PSOrthoTile or PSScene) into an xr.Dataset.

    Args:
        planet_scene_path (Path): path to the Planet Scene
        elevation_path (Path): path to the elevation data
        slope_path (Path): path to the slope data

    Returns:
        xr.Dataset: preprocessed Planet Scene

    """
    # load planet scene
    ds_planet = load_planet_scene(planet_scene_path)

    # calculate xr.dataset ndvi
    ds_ndvi = calculate_ndvi(ds_planet)

    # get xr.dataset for elevation
    ds_elevation = load_auxiliary(planet_scene_path, elevation_path, xr_dataset_name="relative_elevation")

    # get xr.dataset for slope
    ds_slope = load_auxiliary(planet_scene_path, slope_path, xr_dataset_name="slope")

    # # get xr.dataset for tcvis
    # ds_tcvis = load_auxiliary(planet_scene_path, tcvis_path)

    # load udm2
    ds_data_masks = load_data_masks(planet_scene_path)

    # merge to final dataset
    ds_merged = xr.merge([ds_planet, ds_ndvi, ds_elevation, ds_slope, ds_data_masks])

    return ds_merged
