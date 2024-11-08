from upscale import Sentinel2Upscaler
import torch
import time
import logging
from pathlib import Path
import xarray as xr

logger = logging.getLogger(__name__)

def load_s2_scene(fpath: str | Path) -> xr.Dataset:
    """Load a Sentinel 2 satellite GeoTIFF file and return it as an xarray datset.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files.

    Returns:
        xr.Dataset: The loaded dataset

    Raises:
        FileNotFoundError: If no matching TIFF file is found in the specified path.

    """
    start_time = time.time()
    logger.debug(f"Loading Sentinel 2 scene from {fpath}")
    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)

    # Get imagepath
    try:
        s2_image = next(fpath.glob("*_SR_clip.tif"))
    except StopIteration:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath} (.glob('*_SR_clip.tif'))")

    print(s2_image)
    # Define band names and corresponding indices
    s2_da = xr.open_dataarray(s2_image)

    bands = {1: "blue", 2: "green", 3: "red", 4: "nir"}

    # Create a list to hold datasets
    datasets = [
        s2_da.sel(band=index)
        .assign_attrs({"data_source": "s2", "long_name": f"Sentinel 2 {name.capitalize()}"})
        .to_dataset(name=name)
        .drop_vars("band")
        for index, name in bands.items()
    ]

    ds_s2 = xr.merge(datasets)
    logger.debug(f"Loaded Sentinel 2 scene in {time.time() - start_time} seconds.")
    return ds_s2

model_path = "/isipd/projects-noreplica/p_aicore_dev/tohoel001/newmodels/s2super_v1.pt"
#test_img_path = "/isipd/projects-noreplica/p_aicore_dev/darts-nextgen_dev_data/input/sentinel2/20210818T223529_20210818T223531_T03WXP"
test_img_path = "/isipd/projects/p_lucas_chamier/192_60/Test/LR"
img = load_s2_scene(test_img_path)
model = Sentinel2Upscaler(model_path)

upscaled_image = model.upscale_s2_to_planet(img)

print("Shape: ", upscaled_image.shape, upscaled_image.dtype)

