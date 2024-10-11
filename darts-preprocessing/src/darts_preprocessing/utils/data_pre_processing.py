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

    # Find the appropriate planet file
    ps_image = get_planet_imagepath_from_planet_scene_path(planet_scene_path)
    if not ps_image:
        raise FileNotFoundError(f"No matching TIFF files found in {planet_scene_path}")

    # Open the TIFF file using rioxarray

    # Define band names and corresponding indices
    planet_da = rxr.open_rasterio(ps_image)

    bands = {1: "blue", 2: "green", 3: "red", 4: "nir"}

    # Create a list to hold datasets
    datasets = [
        planet_da.sel(band=index).assign_attrs({"data_source": "planet"}).to_dataset(name=name).drop_vars("band")
        for index, name in bands.items()
    ]

    # Merge all datasets into one
    ds_planet = xr.merge(datasets)
    return ds_planet


def calculate_ndvi(planet_scene_dataset: xr.Dataset, nir_band: str = "nir", red_band: str = "red") -> xr.Dataset:
    """Calculate NDVI from an xarray Dataset containing spectral bands.

    Parameters
    ----------
    planet_scene_dataset : xr.Dataset
        The xarray Dataset containing the spectral bands, where the bands are
        indexed along a dimension (e.g., 'band'). The Dataset should have
        dimensions including 'band', 'y', and 'x'.

    nir_band : str, optional
        The name of the NIR band in the Dataset (default is "nir"). This name
        should correspond to the variable name for the NIR band in the 'band'
        dimension.

    red_band : str, optional
        The name of the Red band in the Dataset (default is "red"). This name
        should correspond to the variable name for the Red band in the 'band'
        dimension.

    Returns
    -------
    xr.Dataset
        A new Dataset containing the calculated NDVI values. The resulting
        Dataset will have dimensions (band: 1, y: ..., x: ...) and will be
        named "ndvi".

    Raises
    ------
    ValueError
        If the specified band names do not exist in the provided Dataset.

    Notes
    -----
    NDVI (Normalized Difference Vegetation Index) is calculated using the formula:
        NDVI = (NIR - Red) / (NIR + Red)

    This index is commonly used in remote sensing to assess vegetation health
    and density.

    Example
    -------
    >>> ndvi_data = calculate_ndvi(planet_scene_dataset)

    """
    # Calculate NDVI using the formula
    nir = planet_scene_dataset[nir_band].astype("float32")
    r = planet_scene_dataset[red_band].astype("float32")
    ndvi = (nir - r) / (nir + r)

    return ndvi.assign_attrs({"data_source": "planet"}).to_dataset(name="ndvi")


def geom_from_image_bounds(image_path):
    with rio.open(image_path) as src:
        return [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]


def crs_from_image(image_path):
    with rio.open(image_path) as src:
        return f"EPSG:{src.crs.to_epsg()}"


def resolution_from_image(image_path):
    with rio.open(image_path) as src:
        return src.res


def get_planet_imagepath_from_planet_scene_path(planet_scene_path):
    image_path = list(planet_scene_path.glob("*_SR.tif"))[0]
    return image_path


def get_planet_udm2path_from_planet_scene_path(planet_scene_path):
    udm2_path = list(planet_scene_path.glob("*_udm2.tif"))[0]
    return udm2_path


def load_auxiliary(
    planet_scene_path,
    auxiliary_file_path,
    xr_dataset_name,
    tmp_data_dir=Path("."),
    ds_annotation={"data_source": "ArcticDEM"},
):
    """Load auxiliary raster data by warping it to match the bounds and resolution of a specified Planet scene.

    This function identifies the appropriate Planet scene image file, extracts its bounding box,
    coordinate reference system (CRS), and resolution. It then uses the GDAL `gdalwarp` command to
    warp the auxiliary raster file to match these parameters and returns the resulting raster data as a
    xarray Dataset.

    Parameters
    ----------
    planet_scene_path : Path
        The file path to the directory containing the Planet scene images. The function expects to find
        a TIFF file with a suffix of '_SR.tif'.

    auxiliary_file_path : Path
        The file path to the auxiliary raster file that needs to be warped.

    xr_dataset_name : str
        The name to assign to the resulting xarray Dataset containing the warped data.

    tmp_data_dir : Path, optional
        The directory where the warped output file will be temporarily saved. Defaults to the current
        directory (".").

    ds_annotation : dict, optional
        A dictionary of attributes to assign to the resulting Dataset. Defaults to {"data_source": "ArcticDEM"}.

    Returns
    -------
    xarray.Dataset
        A Dataset containing the warped auxiliary raster data, aligned with the specified Planet scene's
        bounds and resolution.

    Notes
    -----
    This function requires GDAL and Rasterio libraries to be installed and accessible in the Python environment.

    The temporary output file is deleted after loading its data into memory.

    Example
    -------
    >>> data = load_auxiliary(Path('/path/to/planet_scene'), Path('/path/to/auxiliary_file.tif'), 'aux_data')

    """
    with rio.open(get_planet_imagepath_from_planet_scene_path(planet_scene_path)) as ds_planet:
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
    os.remove(outfile)

    return data_array.assign_attrs(ds_annotation).drop_vars("band").to_dataset(name=xr_dataset_name)


def load_data_masks(planet_scene_path):
    """Load valid and quality data masks from a Planet scene.

    This function retrieves the valid and quality data masks from a
    specified Planet scene file. The valid data mask indicates
    areas of valid data (1) versus no data (0). The quality data mask
    assesses the quality of the data, where high quality is marked as
    1 and low quality as 0.

    Parameters
    ----------
    planet_scene_path : str
        The file path to the Planet scene from which to derive the
        masks.

    Returns
    -------
    xarray.Dataset
        A merged xarray Dataset containing two data masks:
        - 'valid_data_mask': A mask indicating valid (1) and no data (0).
        - 'quality_data_mask': A mask indicating high quality (1)
          and low quality (0).

    Notes
    -----
    - The function utilizes the `get_planet_udm2path_from_planet_scene_path`
      to obtain the path to the UDM (User Data Model) file.
    - It uses `rasterio` to read the UDM file and `xarray` for handling
      the datasets.

    """
    udm_path = get_planet_udm2path_from_planet_scene_path(planet_scene_path)
    ds_udm = rxr.open_rasterio(udm_path)

    # valid data mask: valid data = 1, no data = 0
    valid_data_mask = (
        (ds_udm.sel(band=8) == 0)
        .assign_attrs({"data_source": "planet"})
        .to_dataset(name="valid_data_mask")
        .drop_vars("band")
    )

    # quality data mask: high quality = 1, low quality = 0
    quality_data_mask = (
        (ds_udm.sel(band=[2, 3, 4, 5, 6]).max(axis=0) != 1)
        .assign_attrs({"data_source": "planet"})
        .to_dataset(name="quality_data_mask")
    )

    return xr.merge([valid_data_mask, quality_data_mask])
