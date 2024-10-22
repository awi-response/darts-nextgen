import math

import clip
import cv2
import lpips
import numpy as np
import tifffile
import torch

# import open_clip
import torch.nn.functional as F
from torchvision.utils import make_grid

# print("hellow")


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    """Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    # print("tensor type: ", tensor.dtype)
    tensor = tensor.squeeze().cpu()  # .clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        # img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        # print("Tensor: ", tensor.dtype, tensor.shape, tensor.min(), tensor.max())
        img_np = tensor.numpy()
        # print("img_np: ", img_np.dtype, img_np.shape, img_np.min(), img_np.max())
        # img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(f"Only support 4D, 3D and 2D tensor. But received with dimension: {n_dim:d}")
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode="RGB"):
    print("Image shape: ", img.shape)
    tifffile.imwrite(img_path, data=img, metadata={"axes": "CYX"}, imagej=True)
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """Calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def calculate_clipscore(img1, img2, clip_model, **kwargs):
    device = torch.device("cuda")
    # print("calculating clip score")
    # clip_score_full = 0.0
    if clip_model == "clip-ViT-B/16":
        # print(device)
        model, _ = clip.load("ViT-B/16", device=device)
        img_size = (224, 224)

    img1 = img1[:, :, (0, 0, 0)]
    img2 = img2[:, :, (0, 0, 0)]
    tensor1 = torch.as_tensor(img1).permute(2, 0, 1)
    tensor1 = tensor1.unsqueeze(0).to(device).float() / 255
    # print(img2.max(), img2.min())
    tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
    tensor2 = tensor2.unsqueeze(0).to(device).float() / 255

    tensor1 = F.interpolate(tensor1, img_size)
    tensor2 = F.interpolate(tensor2, img_size)

    feats1 = model.encode_image(tensor1)
    feats2 = model.encode_image(tensor2)

    clip_score = F.cosine_similarity(feats1, feats2).detach().item()
    # elif clip_model == 'clipa-ViT-bigG-14':
    #     model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
    #     model = model.to(device)
    #     img_size = (336,336)
    # elif clip_model == 'siglip-ViT-SO400M-14':
    #     model, _, _ = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
    #     model = model.to(device)
    #     img_size = (384,384)
    # else:
    #     print(clip_model, " is not supported for CLIPScore.")

    # Sample different 3 channel images from 4 channel inputs, to simulate RGB.

    # img1_1 = img1[:,:,0:3]
    # img2_1 = img2[:,:,0:3]

    # img1_2 = img1[:,:,[0,1,3]]
    # img2_2 = img2[:,:,[0,1,3]]

    # img1_3 = img1[:,:,[0,2,3]]
    # img2_3 = img2[:,:,[0,2,3]]

    # img1_4 = img1[:,:,[1,2,3]]
    # img2_4 = img2[:,:,[1,2,3]]

    # imgs_1 = [img1_1, img1_2, img1_3, img1_4]
    # imgs_2 = [img2_1, img2_2, img2_3, img2_4]

    # # full_clip_score = torch.zeros()
    # for img1, img2 in zip(imgs_1, imgs_2):
    #     #print(img1.max(), img1.min())
    #     tensor1 = torch.as_tensor(img1).permute(2, 0, 1)
    #     tensor1 = tensor1.unsqueeze(0).to(device).float()/255
    #     #print(img2.max(), img2.min())
    #     tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
    #     tensor2 = tensor2.unsqueeze(0).to(device).float()/255

    #     tensor1 = F.interpolate(tensor1, img_size)
    #     tensor2 = F.interpolate(tensor2, img_size)

    #     feats1 = model.encode_image(tensor1)
    #     feats2 = model.encode_image(tensor2)

    #     clip_score = F.cosine_similarity(feats1, feats2).detach().item()
    #     #print(clip_score)

    #     #print(clip_score.shape)
    #     clip_score_full += clip_score

    # clip_score_full = clip_score_full / 4
    # print(clip_score_full)

    return clip_score


def calculate_lpips(img1, img2, lpips_model, **kwargs):
    device = torch.device("cuda")

    if lpips_model == "alexnet":
        lpips_loss_fn = lpips.LPIPS(net="alex").to(device)  # best forward scores
    elif lpips_model == "vgg":
        lpips_loss_fn = lpips.LPIPS(net="vgg").to(
            device
        )  # closer to "traditional" perceptual loss, when used for optimization

    img1 = img1[:, :, (0, 0, 0)]
    img2 = img2[:, :, (0, 0, 0)]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 = 2 * ((img1 - img1.min()) / (img1.max() - img1.min())) - 1
    img2 = 2 * ((img2 - img2.min()) / (img2.max() - img2.min())) - 1

    tensor1 = torch.as_tensor(img1).permute(2, 0, 1)
    tensor1 = tensor1.unsqueeze(0).to(device)
    # tensor1 = tensor1.unsqueeze(0).to(device).float()/255
    tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
    tensor2 = tensor2.unsqueeze(0).to(device)
    # tensor2 = tensor2.unsqueeze(0).to(device).float()/255
    # print("Minmax: ", tensor1.min(), tensor2.max())

    lpips_loss = lpips_loss_fn(tensor1, tensor2).detach().item()

    return lpips_loss


# print(img1.shape, img2.shape)

# img1_1 = img1[:,:,0:3]
# img2_1 = img2[:,:,0:3]

# img1_2 = img1[:,:,[0,1,3]]
# img2_2 = img2[:,:,[0,1,3]]

# img1_3 = img1[:,:,[0,2,3]]
# img2_3 = img2[:,:,[0,2,3]]

# img1_4 = img1[:,:,[1,2,3]]
# img2_4 = img2[:,:,[1,2,3]]

# imgs_1 = [img1_1, img1_2, img1_3, img1_4]
# imgs_2 = [img2_1, img2_2, img2_3, img2_4]

# lpips_full = 0

# for img1, img2 in zip(imgs_1, imgs_2):
#     #print(img1.shape, img2.shape)
#     tensor1 = torch.as_tensor(img1).permute(2, 0, 1)
#     tensor1 = tensor1.unsqueeze(0).to(device)
#     # tensor1 = tensor1.unsqueeze(0).to(device).float()/255
#     tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
#     tensor2 = tensor2.unsqueeze(0).to(device)
#     #tensor2 = tensor2.unsqueeze(0).to(device).float()/255
#     # print("Minmax: ", tensor1.min(), tensor2.max())

#     lpips_loss = lpips_loss_fn(tensor1, tensor2).detach().item()

#     lpips_full += lpips_loss

# lpips_full = lpips_full/4
# return lpips_full
