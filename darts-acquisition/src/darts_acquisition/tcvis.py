"""Landsat Trends related Data Loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
from pathlib import Path
from typing import Literal

import geopandas as gpd
import odc.geo.crs
import smart_geocubes
import xarray as xr
from odc.geo.geobox import GeoBox
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _sat_to_tcvis_year_lagged(sat_year: int, lag: int) -> Literal[2019, 2020, 2022, 2024]:
    match sat_year + lag:
        case year if year <= 2019:
            tcvis_year = 2019
        case 2020:
            tcvis_year = 2020
        case 2021 | 2022:
            tcvis_year = 2022
        case year if year >= 2023:
            tcvis_year = 2024
        case _:
            raise ValueError(f"Invalid satellite year: {sat_year}. Must be an int.")
    logger.debug(f"Mapping satellite year {sat_year} with lag {lag} to TCVIS year {tcvis_year}.")
    return tcvis_year


def _get_accessor_from_year(
    year: int, lag: int, data_dir: Path | str
) -> smart_geocubes.TCTrend2019 | smart_geocubes.TCTrend2020 | smart_geocubes.TCTrend2022 | smart_geocubes.TCTrend2024:
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    assert ".icechunk" != data_dir.suffix, (
        f"Data directory {data_dir} must not have an .icechunk suffix!"
        " It should point to the directory containing the .icechunk store, not the store itself."
        " Providing the store directly is legacy behaviour!"
    )
    tcvis_year = _sat_to_tcvis_year_lagged(year, lag)
    logger.debug(f"Using TCTrend{tcvis_year} for {year=}")
    data_dir_year = data_dir / f"TCTrend{tcvis_year}.icechunk"
    match tcvis_year:
        case 2019:
            accessor = smart_geocubes.TCTrend2019(data_dir_year, create_icechunk_storage=False, backend="threaded")
        case 2020:
            accessor = smart_geocubes.TCTrend2020(data_dir_year, create_icechunk_storage=False, backend="threaded")
        case 2022:
            accessor = smart_geocubes.TCTrend2022(data_dir_year, create_icechunk_storage=False, backend="threaded")
        case 2024:
            accessor = smart_geocubes.TCTrend2024(data_dir_year, create_icechunk_storage=False, backend="threaded")

    # We want to assume that the datacube is already created to be save in a multi-process environment
    accessor.assert_created()

    return accessor


def create_tcvis_datacubes(years: list[int], data_dir: Path | str, lag: int = 0) -> None:
    """Create the TCVIS datacubes for the given years.

    Should be used in a single-process environment to set up the datacubes for the first time.

    Args:
        years (list[int]): List of years for which to create the datacubes.
        data_dir (Path | str): Path to the directory where the datacubes should be created.
            This should be the directory containing the .icechunk stores, not the stores themselves.
        lag (int, optional): The lag in years to apply when mapping the input years to the TCVIS versions.
            This allows creating TCVIS datacubes that are more appropriate for the satellite data years.
            E.g. for satellite data from 2022, a lag of 1 would create TCVIS datacubes at or after 2023,
            which would be TCTrend2024, while a lag of 0 would create TCTrend2022.
            Defaults to 0.

    Example:
        ```python
        create_tcvis_datacubes(
            years=[2019, 2020, 2022, 2024],
            data_dir="/data/tcvis"
        )
        ```

    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    assert ".icechunk" != data_dir.suffix, (
        f"Data directory {data_dir} must not have an .icechunk suffix!"
        " It should point to the directory containing the .icechunk store, not the store itself."
        " Providing the store directly is legacy behaviour!"
    )

    tcvis_years = {_sat_to_tcvis_year_lagged(year, lag) for year in years}
    logger.info(f"Creating TCVIS datacubes for years {years} (mapped to TCVIS years {tcvis_years}) in {data_dir}.")

    for tcvis_year in tcvis_years:
        data_dir_year = data_dir / f"TCTrend{tcvis_year}.icechunk"
        match tcvis_year:
            case 2019:
                accessor = smart_geocubes.TCTrend2019(data_dir_year, backend="threaded")
            case 2020:
                accessor = smart_geocubes.TCTrend2020(data_dir_year, backend="threaded")
            case 2022:
                accessor = smart_geocubes.TCTrend2022(data_dir_year, backend="threaded")
            case 2024:
                accessor = smart_geocubes.TCTrend2024(data_dir_year, backend="threaded")

        if not accessor.created:
            accessor.create(overwrite=False)


@stopwatch.f("Loading TCVIS", printer=logger.debug, print_kwargs=["year", "data_dir", "lag", "offline"])
def load_tcvis(
    geobox: GeoBox,
    year: int,
    data_dir: Path | str,
    lag: int = 0,
    offline: bool = False,
) -> xr.Dataset:
    """Load TCVIS (Tasseled Cap trends) for the given geobox, fetch new data from GEE if necessary.

    This function loads Tasseled Cap trend data from a local icechunk store. If `offline=False`,
    missing data will be automatically downloaded from Google Earth Engine and stored locally.
    The data contains temporal trends in brightness, greenness, and wetness derived from
    Landsat imagery.

    Args:
        geobox (GeoBox): The geobox for which to load the data. Can be in any CRS.
        year (int): The year for which to load the TCVIS data.
            This is used to determine the relevant time period for the trends.
            As currently only 2019, 2020, 2022 and 2024 TCVIS data is available,
            the year is used to determine the version of the data to load.
        data_dir (Path | str): Path to the icechunk data directory (must have .icechunk suffix).
            This directory stores downloaded TCVIS data for faster consecutive access.
        lag (int, optional): The lag in years to apply when mapping the input year to the TCVIS version.
            This allows loading TCVIS data that is more appropriate for the satellite data year.
            E.g. for satellite data from 2022, a lag of 1 would try to load TCVIS data at or after 2023,
            which would be TCTrend2024, while a lag of 0 would load TCTrend2022.
            Defaults to 1.
        offline (bool, optional): If True, only loads data already present in the local store
            without attempting any downloads. If False, missing data is downloaded from GEE.
            Defaults to False.

    Returns:
        xr.Dataset: The TCVIS dataset with the following data variables:
            - tc_brightness (float): Temporal trend in Tasseled Cap brightness component
            - tc_greenness (float): Temporal trend in Tasseled Cap greenness component
            - tc_wetness (float): Temporal trend in Tasseled Cap wetness component

            The dataset is in the TCVIS native CRS with the buffer applied.
            It is NOT automatically reprojected to match the input geobox's CRS.

    Note:
        The `offline` parameter controls data fetching behavior:

        - When `offline=False`: Uses `smart_geocubes` accessor's `load()` method which automatically
          downloads missing tiles from GEE and persists them to the icechunk store.
        - When `offline=True`: Uses the accessor's `open_xarray()` method to open the existing store
          and crops it to the requested region. Raises an error if data is missing.

        Variable naming: The original TCB_slope, TCG_slope, and TCW_slope variables are renamed
        to follow DARTS conventions (tc_brightness, tc_greenness, tc_wetness).

        Year mapping to TCVIS versions without lag:
        - <= 2019 -> TCTrend2019
        - 2020 -> TCTrend2020
        - 2021 -> TCTrend2022
        - 2022 -> TCTrend2022
        - >= 2023 -> TCTrend2024

        Year mapping to TCVIS versions with lag 1:
        - <= 2018 -> TCTrend2019
        - 2019 -> TCTrend2020
        - 2020 -> TCTrend2022
        - 2021 -> TCTrend2022
        - >= 2022 -> TCTrend2024

    Example:
        Load TCVIS data aligned with optical imagery:

        ```python
        from darts_acquisition import load_tcvis

        # Assume "optical" is a loaded Sentinel-2 dataset
        tcvis = load_tcvis(
            geobox=optical.odc.geobox,
            year=2019,
            data_dir="/data/tcvis.icechunk",
            offline=False
        )

        # Reproject to match optical data's CRS and resolution
        tcvis = tcvis.odc.reproject(optical.odc.geobox, resampling="cubic")
        ```

    """
    accessor = _get_accessor_from_year(year, lag, data_dir)

    assert isinstance(accessor.extent.crs, odc.geo.crs.CRS), "Accessor geobox CRS must be a valid ODC CRS"

    if not offline:
        tcvis = accessor.load(geobox, persist=True)
    else:
        xrcube = accessor.open_xarray()
        reference_geobox = geobox.to_crs(accessor.extent.crs, resolution="same")
        tcvis = xrcube.odc.crop(reference_geobox.extent, apply_mask=False)
        tcvis = tcvis.load()

    # In case there are any missing values, will them with 0
    tcvis = tcvis.fillna(0)

    # Rename to follow our conventions
    tcvis = tcvis.rename_vars(
        {
            "TCB_slope": "tc_brightness",
            "TCG_slope": "tc_greenness",
            "TCW_slope": "tc_wetness",
        }
    )

    return tcvis


@stopwatch.f("Downloading TCVIS", printer=logger.debug, print_kwargs=["year", "lag", "data_dir"])
def download_tcvis(
    aoi: gpd.GeoDataFrame,
    data_dir: Path | str,
    year: int | Literal["all"] | None = None,
    lag: int = 0,
) -> None:
    """Download TCVIS (Tasseled Cap trends) data for the specified area of interest.

    This function downloads Tasseled Cap trend data from Google Earth Engine for the given
    area of interest and stores it in a local icechunk data store for efficient access.

    Args:
        aoi (gpd.GeoDataFrame): Area of interest for which to download TCVIS data.
            Can be in any CRS; will be reprojected to the TCVIS dataset's native CRS.
        data_dir (Path | str): Path to the icechunk data directory (must have .icechunk suffix).
        year (int | Literal["all"], optional): The year for which to download the TCVIS data.
            This is used to determine the relevant time period for the trends and the version of TCVIS to download.
            If "all", downloads all available years (2019, 2020, 2022, 2024).
            If None will try to extract for each aoi the year from a "year" column if it exists,
            otherwise defaults to "all".
            Defaults to None.
        lag (int, optional): The lag in years to apply when mapping the input year to the TCVIS version.
            This allows downloading TCVIS data that is more appropriate for the satellite data year.
            E.g. for satellite data from 2022, a lag of 1 would try to download TCVIS data at or after 2023,
            which would be TCTrend2024, while a lag of 0 would download TCTrend2022.
            Defaults to 0.

    Raises:
        ValueError: If the `year` parameter is not an int, "all", or if the `data_dir` does not have the correct format.

    Note:
        Requires Google Earth Engine authentication to be set up before calling this function.
        Use `ee.Initialize()` or `ee.Authenticate()` as needed.

    Example:
        Download TCVIS for a study area:

        ```python
        import geopandas as gpd
        from shapely.geometry import box
        from darts_acquisition import download_tcvis

        # Define area of interest
        aoi = gpd.GeoDataFrame(
            geometry=[box(-50, 70, -49, 71)],
            crs="EPSG:4326"
        )

        # Download TCVIS
        download_tcvis(
            aoi=aoi,
            data_dir="/data/tcvis.icechunk"
        )
        ```

    """
    logger.debug(f"{year=} {lag=} {'year' in aoi.columns=}")
    match (year, "year" in aoi.columns):
        case (None, True):
            # This groups the AOI by mapped tcvis-years to optimize the download:
            # Instead of triggering a download per year, a download is triggered per tcvis-year,
            # which can cover multiple satellite years, enabling a more efficient use of the threaded download.
            tcvis_years = aoi["year"].apply(lambda y: _sat_to_tcvis_year_lagged(y, lag))
            for tcvis_year in tcvis_years.unique():
                accessor = _get_accessor_from_year(tcvis_year, lag, data_dir)
                aoi_subset = aoi[tcvis_years == tcvis_year]
                accessor.procedural_download(aoi_subset, None)
        case (None, False) | ("all", _):
            years_to_download = [2019, 2020, 2022, 2024]
            for year in years_to_download:
                # Single level recursive call to download each year separately -> Anker is the isinstace call below
                # Will not trigger infinite recursion
                download_tcvis(aoi, data_dir, year, lag)
        case (int(), _):
            accessor = _get_accessor_from_year(year, lag, data_dir)  # ty:ignore[invalid-argument-type]
            accessor.procedural_download(aoi, None)
        case _:
            raise ValueError(f"Invalid year parameter: {year=}. Must be an int, None or 'all'.")
