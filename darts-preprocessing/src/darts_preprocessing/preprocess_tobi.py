"""PLANET scene based preprocessing."""

from pathlib import Path

import xarray as xr

from darts_preprocessing.data_sources.arcticdem import load_arcticdem
from darts_preprocessing.data_sources.planet import load_planet_scene
from darts_preprocessing.engineering.indices import calculate_ndvi
from darts_preprocessing.engineering.quality_masks import load_data_masks


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

    ds_articdem = load_arcticdem(elevation_path, slope_path, ds_planet)

    # # get xr.dataset for tcvis
    # ds_tcvis = load_auxiliary(planet_scene_path, tcvis_path)

    # load udm2
    ds_data_masks = load_data_masks(planet_scene_path)

    # merge to final dataset
    ds_merged = xr.merge([ds_planet, ds_ndvi, ds_articdem, ds_data_masks])

    return ds_merged
