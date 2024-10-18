"""PLANET scene based preprocessing."""

from pathlib import Path

import xarray as xr

from darts_preprocessing.data_sources.arcticdem import load_arcticdem
from darts_preprocessing.data_sources.planet import load_planet_masks, load_planet_scene
from darts_preprocessing.engineering.indices import calculate_ndvi


def load_and_preprocess_planet_scene(planet_scene_path: Path, elevation_path: Path, slope_path: Path) -> xr.Dataset:
    """Load and preprocess a Planet Scene (PSOrthoTile or PSScene) into an xr.Dataset.

    Args:
        planet_scene_path (Path): path to the Planet Scene
        elevation_path (Path): path to the elevation data
        slope_path (Path): path to the slope data

    Returns:
        xr.Dataset: preprocessed Planet Scene

    Examples:
        Data directory structure:

        ```sh
            data/input
            └── planet
                ├── ArcticDEM
                │   ├── relative_elevation
                │   │   └── 4372514_relative_elevation_100.tif
                │   └── slope
                │       └── 4372514_slope.tif
                └── planet
                    └── PSOrthoTile
                        └── 4372514/5790392_4372514_2022-07-16_2459
                            ├── 5790392_4372514_2022-07-16_2459_BGRN_Analytic_metadata.xml
                            ├── 5790392_4372514_2022-07-16_2459_BGRN_DN_udm.tif
                            ├── 5790392_4372514_2022-07-16_2459_BGRN_SR.tif
                            ├── 5790392_4372514_2022-07-16_2459_metadata.json
                            ├── 5790392_4372514_2022-07-16_2459_udm2.tif
        ```

        Load and preprocess a Planet Scene:

        ```python
            from pathlib import Path
            from darts_preprocessing.preprocess import load_and_preprocess_planet_scene

            fpath = Path("data/input/planet/planet/PSOrthoTile/4372514/5790392_4372514_2022-07-16_2459")
            scene_id = fpath.parent.name
            elevation_path = input_data_dir / "ArcticDEM" / "relative_elevation" / f"{scene_id}_relative_elevation_100.tif"
            slope_path = input_data_dir / "ArcticDEM" / "slope" / f"{scene_id}_slope.tif"
            tile = load_and_preprocess_planet_scene(fpath, elevation_path, slope_path)
        ```

    """  # noqa: E501
    # load planet scene
    ds_planet = load_planet_scene(planet_scene_path)

    # calculate xr.dataset ndvi
    ds_ndvi = calculate_ndvi(ds_planet)

    ds_articdem = load_arcticdem(elevation_path, slope_path, ds_planet)

    # # get xr.dataset for tcvis
    # ds_tcvis = load_auxiliary(planet_scene_path, tcvis_path)

    # load udm2
    ds_data_masks = load_planet_masks(planet_scene_path)

    # merge to final dataset
    ds_merged = xr.merge([ds_planet, ds_ndvi, ds_articdem, ds_data_masks])

    return ds_merged
