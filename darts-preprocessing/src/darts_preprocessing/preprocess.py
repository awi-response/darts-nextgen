"""PLANET scene based preprocessing."""

from pathlib import Path

import xarray as xr

from darts_preprocessing.data_sources.arcticdem import load_arcticdem_from_vrt
from darts_preprocessing.data_sources.planet import load_planet_masks, load_planet_scene
from darts_preprocessing.data_sources.s2 import load_s2_masks, load_s2_scene
from darts_preprocessing.data_sources.tcvis import load_tcvis
from darts_preprocessing.engineering.indices import calculate_ndvi


def load_and_preprocess_planet_scene(
    planet_scene_path: Path, slope_vrt: Path, elevation_vrt: Path, cache_dir: Path | None = None
) -> xr.Dataset:
    """Load and preprocess a Planet Scene (PSOrthoTile or PSScene) into an xr.Dataset.

    Args:
        planet_scene_path (Path): path to the Planet Scene
        slope_vrt (Path): path to the ArcticDEM slope VRT file
        elevation_vrt (Path): path to the ArcticDEM elevation VRT file
        cache_dir (Path | None): The cache directory. If None, no caching will be used. Defaults to None.

    Returns:
        xr.Dataset: preprocessed Planet Scene

    Examples:
        ### PS Orthotile

        Data directory structure:

        ```sh
            data/input
            ├── ArcticDEM
            │   ├── elevation.vrt
            │   ├── slope.vrt
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
                        └── 5790392_4372514_2022-07-16_2459_udm2.tif
        ```

        Load and preprocess a Planet Scene:

        ```python
            from pathlib import Path
            from darts_preprocessing.preprocess import load_and_preprocess_planet_scene

            fpath = Path("data/input/planet/PSOrthoTile/4372514/5790392_4372514_2022-07-16_2459")
            arcticdem_dir = input_data_dir / "ArcticDEM"
            tile = load_and_preprocess_planet_scene(fpath, arcticdem_dir / "slope.vrt", arcticdem_dir / "elevation.vrt")
        ```


        ### PS Scene

        Data directory structure:

        ```sh
            data/input
            ├── ArcticDEM
            │   ├── elevation.vrt
            │   ├── slope.vrt
            │   ├── relative_elevation
            │   │   └── 4372514_relative_elevation_100.tif
            │   └── slope
            │       └── 4372514_slope.tif
            └── planet
                └── PSScene
                    └── 20230703_194241_43_2427
                        ├── 20230703_194241_43_2427_3B_AnalyticMS_metadata.xml
                        ├── 20230703_194241_43_2427_3B_AnalyticMS_SR.tif
                        ├── 20230703_194241_43_2427_3B_udm2.tif
                        ├── 20230703_194241_43_2427_metadata.json
                        └── 20230703_194241_43_2427.json
        ```

        Load and preprocess a Planet Scene:

        ```python
            from pathlib import Path
            from darts_preprocessing.preprocess import load_and_preprocess_planet_scene

            fpath = Path("data/input/planet/PSOrthoTile/20230703_194241_43_2427")
            arcticdem_dir = input_data_dir / "ArcticDEM"
            tile = load_and_preprocess_planet_scene(fpath, arcticdem_dir / "slope.vrt", arcticdem_dir / "elevation.vrt")
        ```

    """
    # load planet scene
    ds_planet = load_planet_scene(planet_scene_path)

    # calculate xr.dataset ndvi
    ds_ndvi = calculate_ndvi(ds_planet)

    ds_articdem = load_arcticdem_from_vrt(slope_vrt, elevation_vrt, ds_planet)

    ds_tcvis = load_tcvis(ds_planet, cache_dir)

    # load udm2
    ds_data_masks = load_planet_masks(planet_scene_path)

    # merge to final dataset
    ds_merged = xr.merge([ds_planet, ds_ndvi, ds_articdem, ds_tcvis, ds_data_masks])

    return ds_merged


def load_and_preprocess_sentinel2_scene(
    s2_scene_path: Path, slope_vrt: Path, elevation_vrt: Path, cache_dir: Path | None = None
) -> xr.Dataset:
    """Load and preprocess a Sentinel 2 scene into an xr.Dataset.

    Args:
        s2_scene_path (Path): path to the Sentinel 2 Scene
        slope_vrt (Path): path to the ArcticDEM slope VRT file
        elevation_vrt (Path): path to the ArcticDEM elevation VRT file
        cache_dir (Path | None): The cache directory. If None, no caching will be used. Defaults to None.

    Returns:
        xr.Dataset: preprocessed Sentinel Scene

    Examples:
        Data directory structure:

        ```sh
            data/input
            ├── ArcticDEM
            │   ├── elevation.vrt
            │   ├── slope.vrt
            │   ├── relative_elevation
            │   │   └── 4372514_relative_elevation_100.tif
            │   └── slope
            │       └── 4372514_slope.tif
            └── sentinel2
                └── 20220826T200911_20220826T200905_T17XMJ/
                    ├── 20220826T200911_20220826T200905_T17XMJ_SCL_clip.tif
                    └── 20220826T200911_20220826T200905_T17XMJ_SR_clip.tif
        ```

        Load and preprocess a Sentinel Scene:

        ```python
            from pathlib import Path
            from darts_preprocessing.preprocess import load_and_preprocess_sentinel2_scene

            fpath = Path("data/input/sentinel2/20220826T200911_20220826T200905_T17XMJ")
            arcticdem_dir = input_data_dir / "ArcticDEM"
            tile = load_and_preprocess_planet_scene(fpath, arcticdem_dir / "slope.vrt", arcticdem_dir / "elevation.vrt")
        ```

    """
    # load planet scene
    ds_s2 = load_s2_scene(s2_scene_path)

    # calculate xr.dataset ndvi
    ds_ndvi = calculate_ndvi(ds_s2)

    ds_articdem = load_arcticdem_from_vrt(slope_vrt, elevation_vrt, ds_s2)

    ds_tcvis = load_tcvis(ds_s2, cache_dir)

    # load scl
    ds_data_masks = load_s2_masks(s2_scene_path, ds_s2)

    # merge to final dataset
    ds_merged = xr.merge([ds_s2, ds_ndvi, ds_articdem, ds_tcvis, ds_data_masks])

    return ds_merged
