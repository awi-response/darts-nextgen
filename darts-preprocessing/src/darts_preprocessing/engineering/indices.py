"""Calculation of spectral indices from optical data."""

import logging

import numpy as np
import xarray as xr
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch("Calculating NDVI", printer=logger.debug)
def calculate_ndvi(optical: xr.Dataset) -> xr.Dataset:
    """Calculate NDVI (Normalized Difference Vegetation Index) from spectral bands.

    NDVI is a widely-used vegetation index that indicates photosynthetic activity and
    vegetation health. Values range from -1 to 1, with higher values indicating denser,
    healthier vegetation.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - nir (float32): Near-infrared reflectance [0-1]
            - red (float32): Red reflectance [0-1]

    Returns:
        xr.DataArray: NDVI values with attributes:
            - long_name: "NDVI"

    Note:
        Formula: NDVI = (NIR - Red) / (NIR + Red)

        Input bands are clipped to [0, 1] before calculation to avoid numerical instabilities
        from negative reflectance values or sensor artifacts. The final result is also clipped
        to ensure values remain in the valid [-1, 1] range.

    Example:
        Calculate NDVI from optical data:

        ```python
        from darts_preprocessing import calculate_ndvi

        # optical contains 'nir' and 'red' bands
        ndvi = calculate_ndvi(optical)

        # Mask vegetation
        vegetation_mask = ndvi > 0.3
        ```

    """
    nir = optical["nir"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    ndvi = (nir - r) / (nir + r)
    ndvi = ndvi.clip(-1, 1)
    ndvi = ndvi.assign_attrs({"long_name": "NDVI"}).rename("ndvi")
    return ndvi


@stopwatch("Calculating GNDVI", printer=logger.debug)
def calculate_gndvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate GNDVI (Green Normalized Difference Vegetation Index) from spectral bands.

    GNDVI is similar to NDVI but uses the green band instead of red, making it more sensitive
    to chlorophyll content and useful for mid to late season vegetation monitoring.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - nir (float32): Near-infrared reflectance [0-1]
            - green (float32): Green reflectance [0-1]

    Returns:
        xr.DataArray: GNDVI values with attributes:
            - long_name: "GNDVI"
            - Values clipped to [-1, 1] range

    Note:
        Formula: GNDVI = (NIR - Green) / (NIR + Green)

        Input bands are clipped to [0, 1] to avoid numerical instabilities.

    Example:
        ```python
        from darts_preprocessing import calculate_gndvi

        gndvi = calculate_gndvi(optical)
        ```

    """
    nir = optical["nir"].clip(0, 1)
    g = optical["green"].clip(0, 1)
    gndvi = (nir - g) / (nir + g)
    gndvi = gndvi.clip(-1, 1)
    gndvi = gndvi.assign_attrs({"long_name": "GNDVI"}).rename("gndvi")
    return gndvi


@stopwatch("Calculating GRVI", printer=logger.debug)
def calculate_grvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate GRVI (Green Red Vegetation Index) from spectral bands.

    GRVI uses visible bands to detect vegetation, useful for high-resolution imagery
    where NIR may not be available or for specific vegetation discrimination tasks.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - green (float32): Green reflectance [0-1]
            - red (float32): Red reflectance [0-1]

    Returns:
        xr.DataArray: GRVI values with attributes:
            - long_name: "GRVI"

    Note:
        Formula: GRVI = (Green - Red) / (Green + Red)

        Input bands are clipped to [0, 1] to avoid numerical instabilities.

    References:
        Eng, L.S., Ismail, R., Hashim, W., Baharum, A., 2019.
        The Use of VARI, GLI, and VIgreen Formulas in Detecting Vegetation In aerial Images.
        International Journal of Technology. Volume 10(7), pp. 1385-1394
        https://doi.org/10.14716/ijtech.v10i7.3275

    Example:
        ```python
        from darts_preprocessing import calculate_grvi

        grvi = calculate_grvi(optical)
        ```

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
    """Calculate RVI (Ratio Vegetation Index) from spectral bands.

    RVI is a simple ratio index sensitive to vegetation amount and biomass. Values typically
    range from 0 to over 30 for dense vegetation.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - nir (float32): Near-infrared reflectance [0-1]
            - red (float32): Red reflectance [0-1]

    Returns:
        xr.DataArray: RVI values with attributes:
            - long_name: "RVI"

    Note:
        Formula: RVI = Red / NIR

        Input bands are clipped to [0, 1] to avoid numerical instabilities.

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021

    Example:
        ```python
        from darts_preprocessing import calculate_rvi

        rvi = calculate_rvi(optical)
        ```

    """
    nir = optical["nir"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    rvi = r / nir
    rvi = rvi.assign_attrs({"long_name": "RVI"}).rename("rvi")
    return rvi


@stopwatch("Calculating NRVI", printer=logger.debug)
def calculate_nrvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate NRVI (Normalized Ratio Vegetation Index) from spectral bands.

    NRVI normalizes RVI to a range similar to NDVI, making it more comparable across
    different vegetation densities.

    Args:
        optical (xr.Dataset): Dataset containing:
            - rvi (float32): RVI values (will be calculated if not present)
            - nir, red (float32): Required if RVI not present

    Returns:
        xr.DataArray: NRVI values with attributes:
            - long_name: "NRVI"

    Note:
        Formula: NRVI = (RVI - 1) / (RVI + 1)
        where RVI = Red / NIR

        If RVI is already in the dataset, it will be reused to avoid recalculation.

    Example:
        ```python
        from darts_preprocessing import calculate_nrvi

        nrvi = calculate_nrvi(optical)
        ```

    """
    rvi = optical["rvi"] if "rvi" in optical else calculate_rvi(optical)
    nrvi = (rvi - 1) / (rvi + 1)
    nrvi = nrvi.assign_attrs({"long_name": "NRVI"}).rename("nrvi")
    return nrvi


@stopwatch("Calculating TVI", printer=logger.debug)
def calculate_tvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate TVI (Transformed Vegetation Index) from spectral bands.

    TVI applies a transformation to NDVI to enhance contrast and improve discrimination
    of vegetation conditions.

    Args:
        optical (xr.Dataset): Dataset containing:
            - ndvi (float32): NDVI values (will be calculated if not present)
            - nir, red (float32): Required if NDVI not present

    Returns:
        xr.DataArray: TVI values with attributes:
            - long_name: "TVI"

    Note:
        Formula: TVI = sqrt(NDVI + 0.5)

        If NDVI is already in the dataset, it will be reused to avoid recalculation.

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021

    Example:
        ```python
        from darts_preprocessing import calculate_tvi

        tvi = calculate_tvi(optical)
        ```

    """
    ndvi = optical["ndvi"] if "ndvi" in optical else calculate_ndvi(optical)
    tvi = np.sqrt(ndvi + 0.5)
    tvi = tvi.assign_attrs({"long_name": "TVI"}).rename("tvi")
    return tvi


@stopwatch("Calculating CTVI", printer=logger.debug)
def calculate_ctvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate CTVI (Corrected Transformed Vegetation Index) from spectral bands.

    CTVI is a corrected version of TVI that maintains the sign of the original NDVI values
    while applying the transformation.

    Args:
        optical (xr.Dataset): Dataset containing:
            - ndvi (float32): NDVI values (will be calculated if not present)
            - nir, red (float32): Required if NDVI not present

    Returns:
        xr.DataArray: CTVI values with attributes:
            - long_name: "CTVI"

    Note:
        Formula: CTVI = (NDVI + 0.5) / |NDVI + 0.5| * sqrt(|NDVI + 0.5|)

        If NDVI is already in the dataset, it will be reused to avoid recalculation.

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021

    Example:
        ```python
        from darts_preprocessing import calculate_ctvi

        ctvi = calculate_ctvi(optical)
        ```

    """
    ndvi = optical["ndvi"] if "ndvi" in optical else calculate_ndvi(optical)
    ctvi = (ndvi + 0.5) / np.abs(ndvi + 0.5) * np.sqrt(np.abs(ndvi + 0.5))
    ctvi = ctvi.assign_attrs({"long_name": "CTVI"}).rename("ctvi")
    return ctvi


@stopwatch("Calculating TTVI", printer=logger.debug)
def calculate_ttvi(optical: xr.Dataset) -> xr.DataArray:
    """Calculate TTVI (Thiam's Transformed Vegetation Index) from spectral bands.

    TTVI applies an absolute value transformation to NDVI before the square root,
    making it suitable for both positive and negative NDVI values.

    Args:
        optical (xr.Dataset): Dataset containing:
            - ndvi (float32): NDVI values (will be calculated if not present)
            - nir, red (float32): Required if NDVI not present

    Returns:
        xr.DataArray: TTVI values with attributes:
            - long_name: "TTVI"

    Note:
        Formula: TTVI = sqrt(|NDVI| + 0.5)

        If NDVI is already in the dataset, it will be reused to avoid recalculation.

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021

    Example:
        ```python
        from darts_preprocessing import calculate_ttvi

        ttvi = calculate_ttvi(optical)
        ```

    """
    ndvi = optical["ndvi"] if "ndvi" in optical else calculate_ndvi(optical)
    ttvi = np.sqrt(np.abs(ndvi) + 0.5)
    ttvi = ttvi.assign_attrs({"long_name": "TTVI"}).rename("ttvi")
    return ttvi


@stopwatch("Calculating SAVI", printer=logger.debug)
def calculate_savi(optical: xr.Dataset, s: float = 0.5) -> xr.DataArray:
    """Calculate SAVI (Soil Adjusted Vegetation Index) from spectral bands.

    SAVI minimizes soil brightness influences using a soil-brightness correction factor.
    Useful in areas with sparse vegetation or exposed soil.

    Args:
        optical (xr.Dataset): Dataset containing:
            - ndvi (float32): NDVI values (will be calculated if not present)
            - nir, red (float32): Required if NDVI not present
        s (float, optional): Soil adjustment factor. Common values:
            - 0.5: moderate vegetation cover (default)
            - 0.25: high vegetation cover
            - 1.0: low vegetation cover

    Returns:
        xr.DataArray: SAVI values with attributes:
            - long_name: "SAVI"

    Note:
        Formula: SAVI = NDVI * (1 + s)

    References:
        Lemenkova, Polina.
        "Hyperspectral Vegetation Indices Calculated by Qgis Using Landsat Tm Image: a Case Study of Northern Iceland"
        Advanced Research in Life Sciences, vol. 4, no. 1, Sciendo, 2020, pp. 70-78.
        https://doi.org/10.2478/arls-2020-0021

    Example:
        ```python
        from darts_preprocessing import calculate_savi

        # For sparse vegetation
        savi = calculate_savi(optical, s=1.0)
        ```

    """
    ndvi = optical["ndvi"] if "ndvi" in optical else calculate_ndvi(optical)
    savi = ndvi * (1 + s)
    savi = savi.assign_attrs({"long_name": "SAVI"}).rename("savi")
    return savi


@stopwatch("Calculating EVI", printer=logger.debug)
def calculate_evi(optical: xr.Dataset, g: float = 2.5, c1: float = 6, c2: float = 7.5, l: float = 1) -> xr.DataArray:  # noqa: E741
    """Calculate EVI (Enhanced Vegetation Index) from spectral bands.

    EVI is optimized to enhance vegetation signal with improved sensitivity in high biomass
    regions and improved vegetation monitoring through decoupling of canopy background signal
    and reducing atmospheric influences.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - nir (float32): Near-infrared reflectance [0-1]
            - red (float32): Red reflectance [0-1]
            - blue (float32): Blue reflectance [0-1]
        g (float, optional): Gain factor. Defaults to 2.5.
        c1 (float, optional): Aerosol resistance coefficient for red band. Defaults to 6.
        c2 (float, optional): Aerosol resistance coefficient for blue band. Defaults to 7.5.
        l (float, optional): Canopy background adjustment. Defaults to 1.

    Returns:
        xr.DataArray: EVI values with attributes:
            - long_name: "EVI"

    Note:
        Formula: EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)

        Input bands are clipped to [0, 1] to avoid numerical instabilities.

    References:
        A Huete, K Didan, T Miura, E.P Rodriguez, X Gao, L.G Ferreira,
        Overview of the radiometric and biophysical performance of the MODIS vegetation indices,
        Remote Sensing of Environment, Volume 83, Issues 1-2, 2002, Pages 195-213, ISSN 0034-4257,
        https://doi.org/10.1016/S0034-4257(02)00096-2.

    Example:
        ```python
        from darts_preprocessing import calculate_evi

        evi = calculate_evi(optical)
        ```

    """
    nir = optical["nir"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    b = optical["blue"].clip(0, 1)
    evi = g * (nir - r) / (nir + c1 * r - c2 * b + l)
    evi = evi.assign_attrs({"long_name": "EVI"}).rename("evi")
    return evi


@stopwatch("Calculating VARI", printer=logger.debug)
def calculate_vari(optical: xr.Dataset) -> xr.DataArray:
    """Calculate VARI (Visible Atmospherically Resistant Index) from spectral bands.

    VARI uses only visible bands, designed to minimize atmospheric effects. Useful for
    RGB imagery without NIR band or for atmospheric correction validation.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - green (float32): Green reflectance [0-1]
            - red (float32): Red reflectance [0-1]
            - blue (float32): Blue reflectance [0-1]

    Returns:
        xr.DataArray: VARI values with attributes:
            - long_name: "VARI"

    Note:
        Formula: VARI = (Green - Red) / (Green + Red - Blue)

        Input bands are clipped to [0, 1] to avoid numerical instabilities.

    References:
        Eng, L.S., Ismail, R., Hashim, W., Baharum, A., 2019.
        The Use of VARI, GLI, and VIgreen Formulas in Detecting Vegetation In aerial Images.
        International Journal of Technology. Volume 10(7), pp. 1385-1394
        https://doi.org/10.14716/ijtech.v10i7.3275

    Example:
        ```python
        from darts_preprocessing import calculate_vari

        vari = calculate_vari(optical)
        ```

    """
    g = optical["green"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    b = optical["blue"].clip(0, 1)
    vari = (g - r) / (g + r - b)
    vari = vari.assign_attrs({"long_name": "VARI"}).rename("vari")
    return vari


@stopwatch("Calculating GLI", printer=logger.debug)
def calculate_gli(optical: xr.Dataset) -> xr.DataArray:
    """Calculate GLI (Green Leaf Index) from spectral bands.

    GLI emphasizes green reflectance for vegetation detection using only visible bands.
    Suitable for RGB sensors and aerial imagery.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - green (float32): Green reflectance
            - red (float32): Red reflectance
            - blue (float32): Blue reflectance

    Returns:
        xr.DataArray: GLI values with attributes:
            - long_name: "GLI"

    Note:
        Formula: GLI = (2 * Green - Red - Blue) / (2 * Green + Red + Blue)

    References:
        Eng, L.S., Ismail, R., Hashim, W., Baharum, A., 2019.
        The Use of VARI, GLI, and VIgreen Formulas in Detecting Vegetation In aerial Images.
        International Journal of Technology. Volume 10(7), pp. 1385-1394
        https://doi.org/10.14716/ijtech.v10i7.3275

    Example:
        ```python
        from darts_preprocessing import calculate_gli

        gli = calculate_gli(optical)
        ```

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
    """Calculate TGI (Triangular Greenness Index) from spectral bands.

    TGI is sensitive to chlorophyll content and can estimate leaf area index without
    calibration. Particularly useful for crop monitoring.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - red (float32): Red reflectance [0-1]
            - green (float32): Green reflectance [0-1]
            - blue (float32): Blue reflectance [0-1]

    Returns:
        xr.DataArray: TGI values with attributes:
            - long_name: "TGI"

    Note:
        Formula: TGI = -0.5 * [190 * (Red - Green) - 120 * (Red - Blue)]

        Input bands are clipped to [0, 1] to avoid numerical instabilities.

    References:
        E. Raymond Hunt, Paul C. Doraiswamy, James E. McMurtrey, Craig S.T. Daughtry, Eileen M. Perry, Bakhyt Akhmedov,
        A visible band index for remote sensing leaf chlorophyll content at the canopy scale,
        International Journal of Applied Earth Observation and Geoinformation,
        Volume 21, 2013, Pages 103-112, ISSN 1569-8432,
        https://doi.org/10.1016/j.jag.2012.07.020.

    Example:
        ```python
        from darts_preprocessing import calculate_tgi

        tgi = calculate_tgi(optical)
        ```

    """
    r = optical["red"].clip(0, 1)
    g = optical["green"].clip(0, 1)
    b = optical["blue"].clip(0, 1)
    tgi = -0.5 * (190 * (r - g) - 120 * (r - b))
    tgi = tgi.assign_attrs({"long_name": "TGI"}).rename("tgi")
    return tgi


@stopwatch("Calculating EXG", printer=logger.debug)
def calculate_exg(optical: xr.Dataset) -> xr.DataArray:
    """Calculate EXG (Excess Green Index) from spectral bands.

    EXG highlights green vegetation by emphasizing the green band relative to red and blue.
    Widely used for crop/weed discrimination and precision agriculture.

    Args:
        optical (xr.Dataset): Dataset containing spectral bands:
            - green (float32): Green reflectance [0-1]
            - red (float32): Red reflectance [0-1]
            - blue (float32): Blue reflectance [0-1]

    Returns:
        xr.DataArray: EXG values with attributes:
            - long_name: "EXG"

    Note:
        Formula: EXG = 2 * Green - Red - Blue

        Input bands are clipped to [0, 1] to avoid numerical instabilities.

    References:
        Upendar, K., Agrawal, K.N., Chandel, N.S. et al.
        Greenness identification using visible spectral colour indices for site specific weed management.
        Plant Physiol. Rep. 26, 179-187 (2021).
        https://doi.org/10.1007/s40502-020-00562-0

    Example:
        ```python
        from darts_preprocessing import calculate_exg

        exg = calculate_exg(optical)

        # Threshold for vegetation detection
        vegetation = exg > 0
        ```

    """
    g = optical["green"].clip(0, 1)
    r = optical["red"].clip(0, 1)
    b = optical["blue"].clip(0, 1)
    exg = 2 * g - r - b
    exg = exg.assign_attrs({"long_name": "EXG"}).rename("exg")
    return exg
