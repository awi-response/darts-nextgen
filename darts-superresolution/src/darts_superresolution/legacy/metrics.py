# import os
# import math
# import numpy as np
# import cv2
# import clip
# import torch
# # import open_clip
# import torch.nn.functional as F
# import lpips
# import tifffile

# from torchvision.utils import make_grid

# def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     # print("tensor type: ", tensor.dtype)
#     tensor = tensor.squeeze().cpu()#.clamp_(*min_max)  # clamp
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
#     #tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
#     # print("Tensor shape: ", tensor.shape)
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         #print("This one")
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         #img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 3:
#         #print("Tensor: ", tensor.dtype, tensor.shape, tensor.min(), tensor.max())
#         img_np = tensor.numpy()
#         #print("img_np: ", img_np.dtype, img_np.shape, img_np.min(), img_np.max())
#         #img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     if out_type == np.uint8:
#         img_np = (img_np * 255.0).round()
#         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
#     return img_np.astype(out_type)


# def save_img(img, img_path, mode='RGB'):
#     # print("Image shape: ", img.shape)
#     if img.shape[0] > 1:
#         assert img.shape[0] == len(img_path), "The number of images in the batch must equal the number of paths for the input images."
#         for i in range(0, img.shape[0]):
#             tifffile.imwrite(img_path[i], data=img[i], metadata={"axes":"CYX"}, imagej=True)
#     else:
#         tifffile.imwrite(img_path, data=img, metadata={"axes": "CYX"}, imagej=True)
#     # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#     # cv2.imwrite(img_path, img)


# def calculate_psnr(img1, img2):
#     # img1 and img2 have range [0, 255]
#     if img1.dtype != np.uint8:
#         img1 = np.asarray(((img1 - img1.min()) / (img1.max() - img1.min())) * 255).round().astype(np.uint8)
#     if img2.dtype != np.uint8:
#         img2 = np.asarray(((img2 - img2.min()) / (img2.max() - img2.min())) * 255).round().astype(np.uint8)
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
    
#     # print(20 * np.log10(255.0 / np.sqrt(np.mean((img1 - img2)**2, axis=(1,2,3)))))
#     batch_psnrs = 20 * np.log10(255.0 / np.sqrt(np.mean((img1 - img2)**2, axis=(1,2,3))))
#     mse = np.mean((img1 - img2)**2)

#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse)), batch_psnrs


# def ssim(img1, img2):
# # for inference
#     # img1 = tensor2img(img1)
#     # img2 = tensor2img(img2)

#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()


# def calculate_ssim(img1, img2):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
   
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
     
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     elif img1.ndim == 4:
#         ssims = []
#         for b in range(img1.shape[0]): # cycle through batch
#             channel_ssims = []
#             if img1.shape[1] > 4: # This means that channels are likely last:
#                 for c in range(img1.shape[3]): # cycle through channels
#                     channel_ssims.append(ssim(img1[b,:,:,c], img2[b,:,:,c]))
            
#             else: # Channels come after batch dimension
#                 for c in range(img1.shape[1]): # cycle through channels
#                     channel_ssims.append(ssim(img1[b,c,:,:], img2[b,c,:,:]))
#             # print("SSIM Batch sample: ", b, np.mean(np.asarray(channel_ssims)))
            
#             ssims.append(np.mean(np.asarray(channel_ssims)))
        
#         ssim_batch_mean = np.mean(np.asarray(ssims))

#         return ssim_batch_mean, ssims

#     else:
#         raise ValueError('Wrong input image dimensions.')


# def calculate_clipscore(img1, img2, clip_model, **kwargs):
#     device = torch.device('cuda')
#     #print("calculating clip score")
#     #clip_score_full = 0.0
#     if clip_model == 'clip-ViT-B/16':
#         #print(device)
#         model, _ = clip.load("ViT-B/16", device=device)
#         img_size = (224,224)

#     average_clip_score = 0
#     for c in range(0, img1.shape[0]):
        
#         # print("image shapes before: ", img1.shape, img2.shape)
#         img1_c = np.stack([img1[c],img1[c],img1[c]])
#         img2_c = np.stack([img2[c],img2[c],img2[c]])

#         # print("image shapes: ", img1_c.shape, img2_c.shape)

#         tensor1 = torch.as_tensor(img1_c)#.permute(2, 0, 1)
#         tensor1 = tensor1.unsqueeze(0).to(device).float()#/255
#         #print(img2.max(), img2.min())
#         tensor2 = torch.as_tensor(img2_c)#.permute(2, 0, 1)
#         tensor2 = tensor2.unsqueeze(0).to(device).float()#/255

#         # print("Reshaping tensors: ", tensor1.shape, tensor2.shape)
#         tensor1 = F.interpolate(tensor1, img_size)
#         tensor2 = F.interpolate(tensor2, img_size)

#         feats1 = model.encode_image(tensor1)
#         feats2 = model.encode_image(tensor2)

#         clip_score = F.cosine_similarity(feats1, feats2).detach().item()

#         average_clip_score += clip_score

#     average_clip_score = average_clip_score/img1.shape[0]

#     return clip_score


# def calculate_lpips(img1, img2, lpips_model, **kwargs):
#     device = torch.device('cuda')

#     if lpips_model == 'alexnet':
#         lpips_loss_fn = lpips.LPIPS(net='alex').to(device) # best forward scores
#     elif lpips_model == 'vgg':
#         lpips_loss_fn = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization
    
#     average_lpips_loss = 0
#     # print("Image shape: ", img1.shape)

#     for c in range(0, img1.shape[0]):

#         img1_c = np.stack([img1[c],img1[c],img1[c]])
#         img2_c = np.stack([img2[c],img2[c],img2[c]])

#         if img1_c.min() == img1_c.max() or img2_c.min() == img2_c.max():
#             img_valid = False
#             break
        
#         else:
#             img_valid = True
#         # print("lpips shapes: ", img1_c.shape, img2_c.shape)
#             img1_c = np.asarray(img1_c).astype(np.float32)
#             img2_c = np.asarray(img2_c).astype(np.float32)
#             img1_c = 2*((img1_c-img1_c.min())/(img1_c.max()-img1_c.min()))-1
#             img2_c = 2*((img2_c-img2_c.min())/(img2_c.max()-img2_c.min()))-1

#             tensor1 = torch.as_tensor(img1_c)#.permute(2, 0, 1)
#             tensor1 = tensor1.unsqueeze(0).to(device)
#         # tensor1 = tensor1.unsqueeze(0).to(device).float()/255
#             tensor2 = torch.as_tensor(img2_c)#.permute(2, 0, 1)
#             tensor2 = tensor2.unsqueeze(0).to(device)
#         #tensor2 = tensor2.unsqueeze(0).to(device).float()/255
#         #print("lpips shapes: ", tensor1.shape, tensor2.shape)

#             lpips_loss = lpips_loss_fn(tensor1, tensor2).detach().item()

#             average_lpips_loss += lpips_loss

#     if img_valid:
#         average_lpips_loss = average_lpips_loss/img1.shape[0]
#         return average_lpips_loss
    
#     else:
#         return None

#    # print(img1.shape, img2.shape)

#     # img1_1 = img1[:,:,0:3]
#     # img2_1 = img2[:,:,0:3]

#     # img1_2 = img1[:,:,[0,1,3]]
#     # img2_2 = img2[:,:,[0,1,3]]

#     # img1_3 = img1[:,:,[0,2,3]]
#     # img2_3 = img2[:,:,[0,2,3]]

#     # img1_4 = img1[:,:,[1,2,3]]
#     # img2_4 = img2[:,:,[1,2,3]]

#     # imgs_1 = [img1_1, img1_2, img1_3, img1_4]
#     # imgs_2 = [img2_1, img2_2, img2_3, img2_4]

#     # lpips_full = 0

#     # for img1, img2 in zip(imgs_1, imgs_2):
#     #     #print(img1.shape, img2.shape)
#     #     tensor1 = torch.as_tensor(img1).permute(2, 0, 1)
#     #     tensor1 = tensor1.unsqueeze(0).to(device)
#     #     # tensor1 = tensor1.unsqueeze(0).to(device).float()/255
#     #     tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
#     #     tensor2 = tensor2.unsqueeze(0).to(device)
#     #     #tensor2 = tensor2.unsqueeze(0).to(device).float()/255
#     #     # print("Minmax: ", tensor1.min(), tensor2.max())

#     #     lpips_loss = lpips_loss_fn(tensor1, tensor2).detach().item()

#     #     lpips_full += lpips_loss

#     # lpips_full = lpips_full/4
#     # return lpips_full
