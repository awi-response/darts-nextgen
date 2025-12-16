import numpy as np
from upscale import Sentinel2Upscaler
from data_processing.patching import create_patches_from_tile, create_tile_from_patches
import torch
import matplotlib.pyplot as plt
from math import floor
import tifffile


img = tifffile.imread("/isipd/projects/p_lucas_chamier/192_60/192_60/Input_Channels_nir.tif")

tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
# tensor = torch.rand(1, 4, 200, 200)

N, C, size_y, size_x = tensor.shape
input_patch_size = 120
output_patch_size = 384
stride = 110

patches, non_zero_patches_up, zero_patches_up, nonzero_indices, zero_indices = create_patches_from_tile(tensor, stride, input_patch_size, output_patch_size)

print(patches.shape)
patches_array = patches.numpy()

print(patches_array.shape)

pad_x = (stride - (size_x - input_patch_size) % stride) % stride
pad_y = (stride - (size_y - input_patch_size) % stride) % stride

num_patches_x = floor((size_x + pad_x - input_patch_size) / stride) + 1
num_patches_y = floor((size_y + pad_y - input_patch_size) / stride) + 1

output = create_tile_from_patches(
    patches_array,
    C, 
    input_patch_size, 
    output_patch_size,
    stride,
    num_patches_x,
    num_patches_y, 
    method="random")

print(output.shape)

output = tifffile.imwrite("/isipd/projects/p_lucas_chamier/192_60/192_60/Input_Channels_nir_reconstructed.tif", np.squeeze(output.numpy(), 0), imagej=True)