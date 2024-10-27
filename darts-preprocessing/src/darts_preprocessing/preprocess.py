"""PLANET scene based preprocessing."""

from pathlib import Path

import xarray as xr

from darts_preprocessing.data_sources.arcticdem import load_arcticdem
from darts_preprocessing.data_sources.planet import load_planet_masks, load_planet_scene
from darts_preprocessing.data_sources.tcvis import load_tcvis
from darts_preprocessing.engineering.indices import calculate_ndvi


def load_and_preprocess_planet_scene(
    planet_scene_path: Path, arcticdem_dir: Path, cache_dir: Path | None = None
) -> xr.Dataset:
    """Load and preprocess a Planet Scene (PSOrthoTile or PSScene) into an xr.Dataset.

    Args:
        planet_scene_path (Path): path to the Planet Scene
        arcticdem_dir (Path): path to the ArcticDEM directory
        cache_dir (Path | None): The cache directory. If None, no caching will be used. Defaults to None.

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
            arcticdem_dir = input_data_dir / "ArcticDEM" / "relative_elevation" / f"{scene_id}_relative_elevation_100.tif"
            tile = load_and_preprocess_planet_scene(fpath, arcticdem_dir)
        ```

    """  # noqa: E501
    # load planet scene
    ds_planet = load_planet_scene(planet_scene_path)

    # calculate xr.dataset ndvi
    ds_ndvi = calculate_ndvi(ds_planet)

    ds_articdem = load_arcticdem(arcticdem_dir, ds_planet)

    ds_tcvis = load_tcvis(ds_planet, cache_dir)

    # load udm2
    ds_data_masks = load_planet_masks(planet_scene_path)

    # merge to final dataset
    ds_merged = xr.merge([ds_planet, ds_ndvi, ds_articdem, ds_tcvis, ds_data_masks])

    return ds_merged
