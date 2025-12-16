from upscale import Sentinel2Upscaler
import torch
import time
import logging
from pathlib import Path
import xarray as xr
import tifffile
import numpy as np
from math import floor
from darts_superresolution.data_processing.patching import create_tile_from_patches

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

    # print(s2_image)
    # Define band names and corresponding indices
    s2_da = xr.open_dataarray(s2_image)

    # print(s2_da.shape)

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

    # print("Dataset: ", ds_s2)
    size_x = int(ds_s2.sizes["x"])
    size_y = int(ds_s2.sizes["y"])
    
    ### Patch size 60
    patch_size_s2 = 120
    patch_size_ps = 384
    stride = 120
    # pad_x = (patch_size_s2 - (size_x % patch_size_s2)) % patch_size_s2
    # pad_y = (patch_size_s2 - (size_y % patch_size_s2)) % patch_size_s2

    pad_x = (stride - (size_x - patch_size_s2) % stride) % stride
    pad_y = (stride - (size_y - patch_size_s2) % stride) % stride

    # num_patches_x = int((size_x + pad_x) / patch_size_s2)
    # num_patches_y = int((size_y + pad_y) / patch_size_s2)

    num_patches_x = floor((size_x + pad_x - patch_size_s2) / stride) + 1
    num_patches_y = floor((size_y + pad_y - patch_size_s2) / stride) + 1
#    print("patches num ", num_patches_x, num_patches_y)
    # print("Input shape: ", ds_s2.shape)
    logger.debug(f"Loaded Sentinel 2 scene in {time.time() - start_time} seconds.")
    return num_patches_x, num_patches_y, ds_s2

model_path = "/isipd/projects/p_lucas_chamier/192_60/192_60/Diffusion_Experiment/checkpoints/NewData_shifted_Test_2_Wave_NoWarmRestarts_Continued_best_gen.pth"#"/isipd/projects-noreplica/p_aicore_dev/tohoel001/newmodels/s2super_v1.pt"
test_img_path = "/isipd/projects-noreplica/p_aicore_dev/darts-nextgen_dev_data/input/sentinel2/20210818T223529_20210818T223531_T03WXP"
#test_img_path = "/isipd/projects/p_lucas_chamier/192_60/Test/LR"
num_patches_x, num_patches_y, img = load_s2_scene(test_img_path)

model = Sentinel2Upscaler(model_path)

upscaled_image = model.upscale_s2_to_planet(img)

# print('upscaled_shape: ', upscaled_image.shape)

# upscaled_image_shape = (img.shape[0], img.shape[1], img.shape[2] * patch_size_ps/patch_size_s2, img.shape[3] * patch_size_ps/patch_size_s2)

# upscaled_image = create_tile_from_patches(upscaled_image, 4, upscaled_image.shape[2], num_patches_x, num_patches_y)
upscaled_image_no_overlap = create_tile_from_patches(upscaled_image, 
    4, 
    input_patch_size=120,
    output_patch_size=384,
    stride=120,
    num_patches_x=num_patches_x,
    num_patches_y=num_patches_y, 
    method="average")

print(upscaled_image_no_overlap.shape, upscaled_image_no_overlap.dtype, upscaled_image_no_overlap.min(), upscaled_image_no_overlap.max())

upscaled_image_no_overlap = np.squeeze(np.asarray(upscaled_image_no_overlap), 0).astype(np.uint16)#[:, 0:upscaled_image_shape[2], 0:upscaled_image_shape[3]]
print(upscaled_image_no_overlap.shape, upscaled_image_no_overlap.dtype, upscaled_image_no_overlap.min(), upscaled_image_no_overlap.max())

tifffile.imwrite("/isipd/projects/p_lucas_chamier/192_60/192_60/DDIM_quad_Upsampled_darts_wavelet_color_fix_20steps_no_overlap.tif", upscaled_image_no_overlap.astype("uint16"))

# upscaled_image_average = create_tile_from_patches(upscaled_image, 
#     4, 
#     input_patch_size=120, 
#     output_patch_size=384,
#     stride=110,
#     num_patches_x=num_patches_x,
#     num_patches_y=num_patches_y, 
#     method="average")

# upscaled_image_average = np.squeeze(np.asarray(upscaled_image_average), 0).astype(np.uint16)#[:, 0:upscaled_image_shape[2], 0:upscaled_image_shape[3]]
# tifffile.imwrite("/isipd/projects/p_lucas_chamier/192_60/192_60/Upsampled_darts_no_color_fix_1000steps_average.tif", upscaled_image_average.astype("uint16"))

# upscaled_image_merge = create_tile_from_patches(upscaled_image, 
#     4, 
#     input_patch_size=120, 
#     output_patch_size=384,
#     stride=110,
#     num_patches_x=num_patches_x,
#     num_patches_y=num_patches_y, 
#     method="random")

# upscaled_image_merge = np.squeeze(np.asarray(upscaled_image_merge), 0).astype(np.uint16)#[:, 0:upscaled_image_shape[2], 0:upscaled_image_shape[3]]
# tifffile.imwrite("/isipd/projects/p_lucas_chamier/192_60/192_60/Upsampled_darts_no_color_fix_1000steps_merge.tif", upscaled_image_merge.astype("uint16"))
