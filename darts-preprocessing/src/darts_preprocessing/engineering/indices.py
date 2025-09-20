"""Calculation of spectral indices from optical data."""

import logging

import numpy as np
import xarray as xr
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch("Calculating NDVI", printer=logger.debug)
def calculate_ndvi(optical: xr.Dataset) -> xr.Dataset:
    """Calculate NDVI from an xarray Dataset containing spectral bands.

    This function will clip the NIR and Red bands to the range [0, 1] before calculating NDVI to avoid
    potential numerical instabilities from negative reflections.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated NDVI values.

    Notes:
        NDVI (Normalized Difference Vegetation Index) is calculated using the formula:
            NDVI = (NIR - Red) / (NIR + Red)

    """
    nir = optical["nir"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    ndvi = (nir - r) / (nir + r)
    ndvi = ndvi.clip(-1, 1)
    ndvi = ndvi.assign_attrs({"long_name": "NDVI"}).rename("ndvi")
    return ndvi


@stopwatch("Calculating GNDVI", printer=logger.debug)
def calculate_gndvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate GNDVI (Green Normalized Difference Vegetation Index) from an xarray Dataset containing spectral bands.

    This function will clip the NIR and Green bands to the range [0, 1] before calculating GNDVI to avoid
    potential numerical instabilities from negative reflections.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated GNDVI values.

    Notes:
        GNDVI is calculated using the formula:
            GNDVI = (NIR - Green) / (NIR + Green)

    """
    nir = optical["nir"].clip(0, 1)
    g = optical["green"].clip(0, 1)
    gndvi = (nir - g) / (nir + g)
    gndvi = gndvi.clip(-1, 1)
    gndvi = gndvi.assign_attrs({"long_name": "GNDVI"}).rename("gndvi")
    return gndvi


@stopwatch("Calculating GRVI", printer=logger.debug)
def calculate_grvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate GRVI (Green Red Vegetation Index) from an xarray Dataset containing spectral bands.

    This function will clip the Green and Red bands to the range [0, 1] before calculating GRVI to avoid
    potential numerical instabilities from negative reflections.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated GRVI values.

    Notes:
        GRVI is calculated using the formula:
            GRVI = (Green - Red) / (Green + Red)

    References:
        Eng, L.S., Ismail, R., Hashim, W., Baharum, A., 2019.
        The Use of VARI, GLI, and VIgreen Formulas in Detecting Vegetation In aerial Images.
        International Journal of Technology. Volume 10(7), pp. 1385-1394
        https://doi.org/10.14716/ijtech.v10i7.3275

    """
    g = optical["green"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    grvi = (g - r) / (g + r)
    grvi = grvi.assign_attrs({"long_name": "GRVI"}).rename("grvi")
    return grvi


def calculate_vigreen(optical: xr.Dataset) -> xr.DataArray:
    """Alias for VIGREEN (Vegetation Index Green) from an xarray Dataset containing spectral bands."""  # noqa: DOC201
    logger.warning("VIGREEN is an alias for GRVI, using GRVI calculation.")
    return calculate_grvi(optical)


@stopwatch("Calculating RVI", printer=logger.debug)
def calculate_rvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate RVI (Ratio Vegetation Index) from an xarray Dataset containing spectral bands.

    This function will clip the NIR and Red bands to the range [0, 1] before calculating RVI to avoid
    potential numerical instabilities from negative reflections.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated RVI values.

    Notes:
        RVI is calculated using the formula:
            RVI = NIR / Red

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021

    """
    nir = optical["nir"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    rvi = r / nir
    rvi = rvi.assign_attrs({"long_name": "RVI"}).rename("rvi")
    return rvi


@stopwatch("Calculating NRVI", printer=logger.debug)
def calculate_nrvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate NRVI (Normalized Ratio Vegetation Index) from an xarray Dataset containing spectral bands.

    This will use the RVI if it is already present in the dataset, otherwise it will calculate it first.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated NRVI values.

    Notes:
        NRVI is calculated using the formula:
            NRVI = (RVI - 1) / (RVI + 1)
        where RVI = NIR / Red

    """
    rvi = optical["rvi"] if "rvi" in optical else calculate_rvi(optical)
    nrvi = (rvi - 1) / (rvi + 1)
    nrvi = nrvi.assign_attrs({"long_name": "NRVI"}).rename("nrvi")
    return nrvi


@stopwatch("Calculating TVI", printer=logger.debug)
def calculate_tvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate TVI (Transformed Vegetation Index) from an xarray Dataset containing spectral bands.

    This will use the NDVI if it is already present in the dataset, otherwise it will calculate it first.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated TVI values.

    Notes:
        TVI is calculated using the formula:
            TVI = sqrt(NDVI + 0.5)

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021


    """
    ndvi = optical["ndvi"] if "ndvi" in optical else calculate_ndvi(optical)
    tvi = np.sqrt(ndvi + 0.5)
    tvi = tvi.assign_attrs({"long_name": "TVI"}).rename("tvi")
    return tvi


@stopwatch("Calculating CTVI", printer=logger.debug)
def calculate_ctvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate CTVI (Corrected Transformed Vegetation Index) from an xarray Dataset containing spectral bands.

    This will use the NDVI if it is already present in the dataset, otherwise it will calculate it first.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated CTVI values.

    Notes:
        CTVI is calculated using the formula:
            CTVI = (NDVI + 0.5) / |NDVI + 0.5| * sqrt(|NDVI + 0.5|)

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021


    """
    ndvi = optical["ndvi"] if "ndvi" in optical else calculate_ndvi(optical)
    ctvi = (ndvi + 0.5) / np.abs(ndvi + 0.5) * np.sqrt(np.abs(ndvi + 0.5))
    ctvi = ctvi.assign_attrs({"long_name": "CTVI"}).rename("ctvi")
    return ctvi


@stopwatch("Calculating TTVI", printer=logger.debug)
def calculate_ttvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate TTVI (Thiam's Transformed Vegetation Index) from an xarray Dataset containing spectral bands.

    This will use the NDVI if it is already present in the dataset, otherwise it will calculate it first.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated TTVI values.

    Notes:
        TTVI is calculated using the formula:
            TTVI = sqrt(abs(NDVI) + 0.5)

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021


    """
    ndvi = optical["ndvi"] if "ndvi" in optical else calculate_ndvi(optical)
    ttvi = np.sqrt(np.abs(ndvi) + 0.5)
    ttvi = ttvi.assign_attrs({"long_name": "TTVI"}).rename("ttvi")
    return ttvi


@stopwatch("Calculating SAVI", printer=logger.debug)
def calculate_savi(optical: xr.Dataset, s: float = 0.5) -> xr.DataArray:
    """Calculate SAVI (Soil Adjusted Vegetation Index) from an xarray Dataset containing spectral bands.

    This will use the NDVI if it is already present in the dataset, otherwise it will calculate it first.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.
        s (float): The soil adjustment factor.

    Returns:
        xr.DataArray: A new DataArray containing the calculated SAVI values.

    Notes:
        SAVI is calculated using the formula:
            SAVI = NDVI * (1 + s)

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021


    """
    ndvi = optical["ndvi"] if "ndvi" in optical else calculate_ndvi(optical)
    savi = ndvi * (1 + s)
    savi = savi.assign_attrs({"long_name": "SAVI"}).rename("savi")
    return savi


@stopwatch("Calculating EVI", printer=logger.debug)
def calculate_evi(optical: xr.Dataset, g: float = 2.5, c1: float = 6, c2: float = 7.5, l: float = 1) -> xr.DataArray:  # noqa: E741
    """Calculate EVI (Enhanced Vegetation Index) from an xarray Dataset containing spectral bands.

    This function will clip the optical bands to the range [0, 1] before calculating VARI to avoid
    potential numerical instabilities from negative reflections.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.
        g (float): Gain factor (default: 2.5).
        c1 (float): Aerosol resistance coefficient for the red band (default: 6).
        c2 (float): Aerosol resistance coefficient for the blue band (default: 7.5).
        l (float): Canopy background adjustment (default: 1).

    Returns:
        xr.DataArray: A new DataArray containing the calculated EVI values.

    Notes:
        EVI is calculated using the formula:
            EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)

    References:
        A Huete, K Didan, T Miura, E.P Rodriguez, X Gao, L.G Ferreira,
        Overview of the radiometric and biophysical performance of the MODIS vegetation indices,
        Remote Sensing of Environment, Volume 83, Issues 1-2, 2002, Pages 195-213, ISSN 0034-4257,
        https://doi.org/10.1016/S0034-4257(02)00096-2.

    """
    nir = optical["nir"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    b = optical["blue"].clip(0, 1)
    evi = g * (nir - r) / (nir + c1 * r - c2 * b + l)
    evi = evi.assign_attrs({"long_name": "EVI"}).rename("evi")
    return evi


@stopwatch("Calculating VARI", printer=logger.debug)
def calculate_vari(optical: xr.Dataset) -> xr.DataArray:
    """Calculate VARI (Visible Atmospherically Resistant Index) from an xarray Dataset containing spectral bands.

    This function will clip the optical bands to the range [0, 1] before calculating VARI to avoid
    potential numerical instabilities from negative reflections.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated VARI values.

    Notes:
        VARI is calculated using the formula:
            VARI = (Green - Red) / (Green + Red - Blue)

    References:
        Eng, L.S., Ismail, R., Hashim, W., Baharum, A., 2019.
        The Use of VARI, GLI, and VIgreen Formulas in Detecting Vegetation In aerial Images.
        International Journal of Technology. Volume 10(7), pp. 1385-1394
        https://doi.org/10.14716/ijtech.v10i7.3275

    """
    g = optical["green"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    b = optical["blue"].clip(0, 1)
    vari = (g - r) / (g + r - b)
    vari = vari.assign_attrs({"long_name": "VARI"}).rename("vari")
    return vari


@stopwatch("Calculating GLI", printer=logger.debug)
def calculate_gli(optical: xr.Dataset) -> xr.DataArray:
    """Calculate GLI (Green Leaf Index) from an xarray Dataset containing spectral bands.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated GLI values.

    Notes:
        GLI is calculated using the formula:
            GLI = (2 x Green - Red - Blue) / (2 x Green + Red + Blue)

    References:
        Eng, L.S., Ismail, R., Hashim, W., Baharum, A., 2019.
        The Use of VARI, GLI, and VIgreen Formulas in Detecting Vegetation In aerial Images.
        International Journal of Technology. Volume 10(7), pp. 1385-1394
        https://doi.org/10.14716/ijtech.v10i7.3275

    """
    g = optical["green"]
    r = optical["red"]
    b = optical["blue"]
    gli = (2 * g - r - b) / (2 * g + r + b)
    gli = gli.assign_attrs({"long_name": "GLI"}).rename("gli")
    return gli


def calculate_vdvi(optical: xr.Dataset) -> xr.DataArray:
    """Alias for GLI (Green Leaf Index) from an xarray Dataset containing spectral bands."""  # noqa: DOC201
    logger.warning("VDVI is an alias for GLI, using GLI calculation.")
    return calculate_gli(optical)


@stopwatch("Calculating TGI", printer=logger.debug)
def calculate_tgi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate TGI (Triangular Greenness Index) from an xarray Dataset containing spectral bands.

    This function will clip the optical bands to the range [0, 1] before calculating TGI to avoid
    potential numerical instabilities from negative reflections.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated TGI values.

    Notes:
        TGI is calculated using the formula:
            TGI = -0.5 x [190 x (Red - Green) - 120 x (Red - Blue)]

    References:
        E. Raymond Hunt, Paul C. Doraiswamy, James E. McMurtrey, Craig S.T. Daughtry, Eileen M. Perry, Bakhyt Akhmedov,
        A visible band index for remote sensing leaf chlorophyll content at the canopy scale,
        International Journal of Applied Earth Observation and Geoinformation,
        Volume 21, 2013, Pages 103-112, ISSN 1569-8432,
        https://doi.org/10.1016/j.jag.2012.07.020.

    """
    r = optical["red"].clip(0, 1)
    g = optical["green"].clip(0, 1)
    b = optical["blue"].clip(0, 1)
    tgi = -0.5 * (190 * (r - g) - 120 * (r - b))
    tgi = tgi.assign_attrs({"long_name": "TGI"}).rename("tgi")
    return tgi


@stopwatch("Calculating EXG", printer=logger.debug)
def calculate_exg(optical: xr.Dataset) -> xr.DataArray:
    """Calculate EXG (Excess Green Index) from an xarray Dataset containing spectral bands.

    This function will clip the optical bands to the range [0, 1] before calculating EXG to avoid
    potential numerical instabilities from negative reflections.

    Args:
        optical (xr.Dataset): The xarray Dataset containing the spectral bands.

    Returns:
        xr.DataArray: A new DataArray containing the calculated EXG values.

    Notes:
        EXG is calculated using the formula:
            EXG = 2 x Green - Red - Blue

    References:
        Upendar, K., Agrawal, K.N., Chandel, N.S. et al.
        Greenness identification using visible spectral colour indices for site specific weed management.
        Plant Physiol. Rep. 26, 179-187 (2021).
        https://doi.org/10.1007/s40502-020-00562-0

    """
    g = optical["green"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    b = optical["blue"].clip(0, 1)
    exg = 2 * g - r - b
    exg = exg.assign_attrs({"long_name": "EXG"}).rename("exg")
    return exg
