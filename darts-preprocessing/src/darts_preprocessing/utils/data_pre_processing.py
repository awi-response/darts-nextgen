import os
from pathlib import Path

import rasterio as rio
import rioxarray as rxr
import xarray as xr


# Preprocess data
def load_planet_scene(planet_scene_path: str | Path) -> xr.Dataset:
    """Load a PlanetScope satellite TIFF file and return it as an xarray dataset.

    Parameters
    ----------
    planet_scene_path (Union[str, Path]): The path to the directory containing the TIFF files
                                           or a specific path to the TIFF file.

    Returns
    -------
    xr.Dataset: The loaded dataset or raises an error if no valid TIFF file is found.

    Raises
    ------
    FileNotFoundError: If no matching TIFF file is found in the specified path.

    """
    # Convert to Path object if a string is provided
    if isinstance(planet_scene_path, str):
        planet_scene_path = Path(planet_scene_path)

    # Find the appropriate TIFF file
    ps_image = list(planet_scene_path.glob(f"{planet_scene_path.name}_*_SR.tif"))

    if not ps_image:
        raise FileNotFoundError(f"No matching TIFF files found in {planet_scene_path}")

    # Open the TIFF file using rioxarray
    return rxr.open_rasterio(ps_image[0])


def calculate_ndvi(planet_scene_dataarray: xr.DataArray, nir_band: int = 4, red_band: int = 3) -> xr.Dataset:
    """Calculate NDVI from an xarray DataArray containing spectral bands.

    Parameters
    ----------
    planet_scene_dataarray : xr.DataArray
        The xarray DataArray containing the spectral bands, where the bands are
        indexed along a dimension (e.g., 'band').

    nir_band : int, optional
        The index of the NIR band in the DataArray (default is 4).

    red_band : int, optional
        The index of the Red band in the DataArray (default is 3).

    Returns
    -------
    xr.DataArray
        A new DataArray containing the calculated NDVI values.

    Raises
    ------
    ValueError
        If the specified band indices are out of bounds for the provided DataArray.

    Notes
    -----
    NDVI is calculated using the formula:
        NDVI = (NIR - Red) / (NIR + Red)

    """
    # Calculate NDVI using the formula
    nir = planet_scene_dataarray.sel(band=nir_band).astype("float32")
    r = planet_scene_dataarray.sel(band=red_band).astype("float32")
    ndvi = (nir - r) / (nir + r)

    return ndvi


def geom_from_image_bounds(image_path):
    with rio.open(image_path) as src:
        return [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]


def crs_from_image(image_path):
    with rio.open(image_path) as src:
        return f"EPSG:{src.crs.to_epsg()}"


def resolution_from_image(image_path):
    with rio.open(image_path) as src:
        return src.res


def load_auxiliary(planet_scene_path, auxiliary_file_path, tmp_data_dir=Path(".")):
    """Load auxiliary raster data by warping it to match the bounds and resolution of a specified Planet scene.

    This function identifies the appropriate Planet scene image file, extracts its bounding box,
    coordinate reference system (CRS), and resolution. It then uses the GDAL `gdalwarp` command to
    warp the auxiliary raster file to match these parameters and returns the resulting raster data as a
    NumPy array.

    Parameters
    ----------
    planet_scene_path : Path
        The file path to the directory containing the Planet scene images. The function expects to find
        a TIFF file with a suffix of '_SR.tif'.

    auxiliary_file_path : Path
        The file path to the auxiliary raster file that needs to be warped.

    tmp_data_dir : Path, optional
        The directory where the warped output file will be temporarily saved. Defaults to the current
        directory (".").

    Returns
    -------
    data_array : xarray.DataArray
        A DataArray containing the warped auxiliary raster data, aligned with the specified Planet scene's
        bounds and resolution.

    Notes
    -----
    This function requires GDAL and Rasterio libraries to be installed and accessible in the Python environment.

    The temporary output file is deleted after loading its data into memory.

    Example:
    --------
    >>> data = load_auxiliary(Path('/path/to/planet_scene'), Path('/path/to/auxiliary_file.tif'))

    """
    with rio.open(planet_scene_path) as ds_planet:
        bbox = ds_planet.bounds
        crs = ds_planet.crs
        res_x, res_y = ds_planet.res

    outfile = tmp_data_dir / "el.tif"

    # setup and run export
    s_warp = f"gdalwarp -te {bbox.left} {bbox.bottom} {bbox.right} {bbox.top} -r cubic -tr {res_x} {res_y} -t_srs {crs} {auxiliary_file_path} {outfile}"
    # print(s_warp)
    os.system(s_warp)

    # load elevation layer
    data_array = rxr.open_rasterio(outfile)
    # delete temporarary file
    # outfile.unlink()
    os.remove(outfile)

    return data_array
