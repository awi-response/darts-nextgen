"""Patching methods for inference."""

import logging
import torch
import torch.nn.functional as F  # noqa: N812
import numpy as np
import tifffile
import tqdm as tqdm
import warnings
from darts_superresolution.data_processing.util import transform_augment_tensor

logger = logging.getLogger(__name__)


def create_patches_from_tile(
    input_tensor: torch.Tensor, 
    stride: int, 
    input_patch_size: int, 
    output_patch_size: int) -> torch.Tensor:
    """Generate patches from a tile.

    Args:
        input_tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The patches.

    """
    logger.debug(f"Creating patched from tile of shape {input_tensor.shape}")

    ## s2 refers to the initial patch size
    patch_size_s2 = input_patch_size #60

    ## ps refers to the upsampled patch size
    patch_size_ps = output_patch_size #283

    ## We calculate the required padding in both dimensions
    _n, _c, size_x, size_y = input_tensor.size()

    pad_x = (stride - (size_x - patch_size_s2) % stride) % stride
    pad_y = (stride - (size_y - patch_size_s2) % stride) % stride
    
    logger.debug(f"Padding the input tensor with {pad_x} on the x-axis and {pad_y} on the y-axis")

    input_tensor = F.pad(input_tensor, (pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2), mode="reflect")

    logger.debug(f"Padded tensor shape: {input_tensor.shape}")
    
    ## We divide the tensor into patches, with stride equal to patch size and collect the patches in a new tensor: patches
    N, C, H, W = input_tensor.shape
    print("input patch size: ", input_tensor.shape)
    unfold = torch.nn.Unfold(kernel_size=patch_size_s2, stride=stride) # stride=patch_size_s2
    patches = unfold(input_tensor)  # shape: (N, C * patch_h * patch_w, num_patches)
    patches = patches.transpose(1, 2)  # (N, num_patches, patch_size^2 * C)
    patches = patches.view(-1, C, patch_size_s2, patch_size_s2)

    logger.debug(f"Generated patches of shape {patches.shape}")

    ## We upsample the patches to the new patch size ps
    print("PlanetScope: ", patch_size_ps)
    upsample_patches = torch.nn.Upsample(size=(patch_size_ps, patch_size_ps), mode="bicubic")

    patches_up = upsample_patches(patches)

    ## We find the patches which are zero across all channels
    is_zero = patches_up.abs().sum(dim=(1, 2, 3)) == 0

    ## collect the tensors with zero and nonzero patches
    zero_patches_up = patches_up[is_zero]
    nonzero_patches_up = patches_up[~is_zero]

    ## We find the indices of the zero and nonzero patches
    zero_indices = torch.nonzero(is_zero, as_tuple=False).squeeze(1)      # shape: [Z]
    nonzero_indices = torch.nonzero(~is_zero, as_tuple=False).squeeze(1)
    
    logger.debug(f"Upsampled patches of shape {patches_up.shape}")

    ## We use a simple transformation to bring the channels into the right range
    [non_zero_patches_up] = transform_augment_tensor([nonzero_patches_up], split="val", min_max=(-1, 1))

    logger.debug(f"Transformed patches of shape {non_zero_patches_up[0].shape}")
    
    return patches_up, non_zero_patches_up, zero_patches_up, nonzero_indices, zero_indices

def is_integer(val, eps=1e-6):
    return abs(val - round(val)) < eps

# Stitching back together

def create_tile_from_patches(
    input_array: np.ndarray, 
    channels: int, 
    input_patch_size: int, 
    output_patch_size: int,
    stride: int,
    num_patches_x: int,
    num_patches_y: int, 
    method="average") -> np.array:
    
    """Generate tile from patches.

    Args:
        input_tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The patches.

    """
    # We convert the input patches (input array) which are the upsampled patches from create_patches_from_tile into tensor.
    
    _, C, _, _ = input_array.shape
    # We create an empty array to fill with the new patches

    upsample_factor = output_patch_size / input_patch_size
    output_stride = stride * upsample_factor

    if not is_integer(output_stride):
        warnings.warn(
            f"Invalid output stride: output_stride = {output_stride:.4f} is not an integer.\n"
            f"Ensure output_patch_size / input_patch_size * stride is an integer."
        )
        breakpoint()
    else:
        output_stride = int(round(output_stride))

    # === Create Output Tensor and Overlap Counters ===
    height = (num_patches_y - 1) * output_stride + output_patch_size
    width  = (num_patches_x - 1) * output_stride + output_patch_size
    
    output = np.zeros((1, channels, height, width), dtype=input_array.dtype)
    if method == "average":
        overlap_counter = np.zeros_like(output)
    else:
        stack = np.zeros((1, C, height, width), dtype=input_array.dtype)
        pixel_stack = [[[] for _ in range(width)] for _ in range(height)]

    # We iteratively fill the new array with the patches to create the full array.
    patch_idx = 0    
    for j in tqdm.tqdm(range(num_patches_y)):
        for i in range(num_patches_x):
            top = j * output_stride
            left = i * output_stride
            patch = input_array[patch_idx]
            if method == "average":
                output[:, :, top:top+output_patch_size, left:left+output_patch_size] += patch#torch.from_numpy(patch)
                overlap_counter[:, :, top:top+output_patch_size, left:left+output_patch_size] += 1

            elif method == "random":
                for y in range(output_patch_size):
                    for x in range(output_patch_size):
                        pixel_stack[top + y][left + x].append(patch[:,y,x])

            else:
                raise ValueError(f"Unsupported method: {method}")
            
            patch_idx += 1

    if method == "average":
        output = output / np.clip(overlap_counter, a_min=1, a_max=None)
        output = torch.from_numpy(output)
        
    elif method == "random":
        for h in range(height):
            for w in range(width):
                values = pixel_stack[h][w]
                if values:
                    idx = np.random.randint(len(values))
                    stack[:,:,h,w] = values[idx]
        
        output = torch.from_numpy(stack)

    return output
