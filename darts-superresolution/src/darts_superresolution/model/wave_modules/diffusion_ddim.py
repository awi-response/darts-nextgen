# import logging
# import math
# from functools import partial
# from inspect import isfunction

# import numpy as np
# import torch
# import torch.nn.functional as F
# from pytorch_wavelets import DWTForward, DWTInverse
# from torch import nn
# from tqdm import tqdm

# logger = logging.getLogger(__name__)


# def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
#     betas = linear_end * np.ones(n_timestep, dtype=np.float64)
#     warmup_time = int(n_timestep * warmup_frac)
#     betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
#     return betas


# def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
#     if schedule == "quad":
#         betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
#     elif schedule == "linear":
#         betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
#     elif schedule == "warmup10":
#         betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
#     elif schedule == "warmup50":
#         betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
#     elif schedule == "const":
#         betas = linear_end * np.ones(n_timestep, dtype=np.float64)
#     elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
#         betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
#     elif schedule == "cosine":
#         timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
#         alphas = timesteps / (1 + cosine_s) * math.pi / 2
#         alphas = torch.cos(alphas).pow(2)
#         alphas = alphas / alphas[0]
#         betas = 1 - alphas[1:] / alphas[:-1]
#         betas = betas.clamp(max=0.999)
#     else:
#         raise NotImplementedError(schedule)
#     return betas


# # gaussian diffusion trainer class


# def exists(x):
#     return x is not None


# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d


# class DWSR(nn.Module):
#     def __init__(self, in_channels, features, out_channels, kernel_size=3, padding=1, stride=1, groups=1, depth=10):
#         super(DWSR, self).__init__()

#         self.conv_layers = [
#             nn.Conv2d(
#                 in_channels,
#                 features,
#                 kernel_size=kernel_size,
#                 padding=padding,
#                 stride=stride,
#                 bias=False,
#                 groups=groups,
#             ),
#             nn.ReLU(),
#         ]
#         for i in range(depth):
#             self.conv_layers.append(
#                 nn.Conv2d(
#                     features,
#                     features,
#                     kernel_size=kernel_size,
#                     padding=padding,
#                     stride=stride,
#                     bias=False,
#                     groups=groups
#                 )
#             )
#             self.conv_layers.append(nn.ReLU())
#         self.conv_layers.append(
#             nn.Conv2d(
#                 features,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 padding=padding,
#                 stride=stride,
#                 bias=False,
#                 groups=groups
#             )
#         )
#         self.convs = nn.Sequential(*self.conv_layers)

#     def forward(self, x):
#         # print("forward: ", x.shape)
#         output = self.convs(x) + x
#         # print("after forward: ", output.shape)
#         return output


# class GaussianDiffusion(nn.Module):
#     def __init__(self, denoise_fn, image_size, channels=4, loss_type="l1", conditional=True, schedule_opt=None):
#         super().__init__()
#         self.channels = channels
#         self.image_size = image_size
#         self.denoise_fn = denoise_fn
#         self.dwsr = DWSR(16, 64, 16, depth=10)
#         self.loss_type = loss_type
#         self.conditional = conditional
#         self.xfm = DWTForward(J=1, mode="zero", wave="haar")
#         self.ifm = DWTInverse(mode="zero", wave="haar")
#         if schedule_opt is not None:
#             pass
#             # self.set_new_noise_schedule(schedule_opt)

#         # print("Channels: ", self.channels)
#         # print("Image Size: ", self.image_size)

#     def set_loss(self, device):
#         if self.loss_type == "l1":
#             self.loss_func = nn.L1Loss(reduction="sum").to(device)
#         elif self.loss_type == "l2":
#             self.loss_func = nn.MSELoss(reduction="sum").to(device)
#         else:
#             raise NotImplementedError()

#     def set_new_noise_schedule(self, schedule_opt, device):
#         to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

#         betas = make_beta_schedule(
#             schedule=schedule_opt["schedule"],
#             n_timestep=schedule_opt["n_timestep"],
#             linear_start=schedule_opt["linear_start"],
#             linear_end=schedule_opt["linear_end"],
#         )
#         betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
#         alphas = 1.0 - betas
#         alphas_cumprod = np.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
#         self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

#         (timesteps,) = betas.shape
#         self.num_timesteps = int(timesteps)
#         self.register_buffer("betas", to_torch(betas))
#         self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
#         self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
#         self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
#         self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
#         self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
#         self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         posterior_variance = betas
#         # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
#         self.register_buffer("posterior_variance", to_torch(posterior_variance))
#         # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
#         self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
#         self.register_buffer(
#             "posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
#         )
#         self.register_buffer(
#             "posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
#         )

#     def apply_dwt(self, images):
#         # print("dwt images: ", images.shape)
#         images_LL, hfreq_tuple = self.xfm(images)

#         return torch.cat(
#             [images_LL, hfreq_tuple[0][:, :, 0, :, :], hfreq_tuple[0][:, :, 1, :, :], hfreq_tuple[0][:, :, 2, :, :]], 1
#         )

#     def apply_idwt(self, features, targetW, targetH):
#         sr_images_LL = features[:, 0:4, :, :]#changed it from 0:3
#         sr_images_HL = features[:, 4:8, :, :].unsqueeze(2)
#         sr_images_LH = features[:, 8:12, :, :].unsqueeze(2)
#         sr_images_HH = features[:, 12:16, :, :].unsqueeze(2)

#         sr_HFreqs = torch.cat([sr_images_HL, sr_images_LH, sr_images_HH], 2)
#         sr_images = self.ifm((sr_images_LL, [sr_HFreqs]))
#         sr_images = F.interpolate(sr_images, size=(targetW, targetH), mode="bicubic")

#         return sr_images

#     def predict_start_from_noise(self, x_t, t, noise):
#         return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

#     def q_posterior(self, x_start, x_t, t):
#         posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
#         posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
#         return posterior_mean, posterior_log_variance_clipped

#     def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
#         batch_size = x.shape[0]
#         noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
#         if condition_x is not None:
#             x_recon = self.predict_start_from_noise(
#                 x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
#             )
#         else:
#             x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, noise_level))

#         if clip_denoised:
#             x_recon.clamp_(-1.0, 1.0)

#         model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
#         return model_mean, posterior_log_variance

#     # ===== ORIGINAL DDPM SAMPLING =====
#     @torch.no_grad()
#     def p_sample(self, x, t, clip_denoised=True, condition_x=None):
#         model_mean, model_log_variance = self.p_mean_variance(
#             x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x
#         )
#         noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
#         return model_mean + noise * (0.5 * model_log_variance).exp()

#     @torch.no_grad()
#     def p_sample_loop(self, x_in, continous=False, sample_inter=None):
#         """Original DDPM sampling loop - keeps all timesteps"""
#         _, _, w, h = x_in.shape
#         x_in = self.apply_dwt(x_in)
#         device = self.betas.device
#         if sample_inter is None:
#             sample_inter = 1 | (self.num_timesteps // 10)
#         x_sr = self.dwsr(x_in)
#         if not self.conditional:
#             shape = x_in
#             img = torch.randn(shape, device=device)
#             ret_img = img
#             for i in tqdm(
#                 reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps
#             ):
#                 img = self.p_sample(img, i)
#                 if i % sample_inter == 0:
#                     ret_img = torch.cat([ret_img, img], dim=0)
#         else:
#             x = x_in
#             shape = x.shape
#             img = torch.randn(shape, device=device)
#             ret_img = x
#             for i in tqdm(
#                 reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps
#             ):
#                 img = self.p_sample(img, i, condition_x=x)
#                 if i % sample_inter == 0:
#                     ret_img = torch.cat([ret_img, img], dim=0)
#         if continous:
#             result = self.apply_idwt((ret_img[0]).unsqueeze(0), w, h)
#             result = torch.cat([result, self.apply_idwt((ret_img[0] + x_sr[0]).unsqueeze(0), w, h)], 0)
#             for i in range(1, len(ret_img)):
#                 result = torch.cat([result, self.apply_idwt((ret_img[i] + x_sr[0]).unsqueeze(0), w, h)], 0)
#             return result
#         else:
#             # print("SR: ", x_sr.shape, "Ret_img: ", ret_img[-16:].shape)
#             result = self.apply_idwt((ret_img[-x_sr.shape[0]:] + x_sr), w, h)
#             return result

#     # ===== NEW DDIM SAMPLING METHODS =====
#     @torch.no_grad()
#     def ddim_step(self, x, t, t_next, clip_denoised=True, condition_x=None, eta=0.0):
#         """
#         Perform one DDIM step from timestep t to timestep t_next.
        
#         Args:
#             x: current noisy image
#             t: current timestep
#             t_next: next timestep (should be < t)
#             clip_denoised: whether to clip predicted x_0
#             condition_x: conditioning input (for conditional generation)
#             eta: amount of stochasticity (0 = deterministic DDIM, 1 = DDPM)
#         """
#         batch_size = x.shape[0]
        
#         # Get noise level for current timestep
#         noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        
#         # Predict noise using the denoising network
#         if condition_x is not None:
#             predicted_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
#         else:
#             predicted_noise = self.denoise_fn(x, noise_level)
        
#         # Predict x_0 from current x_t and predicted noise
#         alpha_cumprod_t = self.alphas_cumprod[t]
#         alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0).to(x.device)
        
#         sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
#         sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
#         # Predict original sample (x_0)
#         pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        
#         if clip_denoised:
#             pred_x0 = pred_x0.clamp(-1.0, 1.0)
        
#         # Compute coefficients for x_{t-1}
#         sqrt_alpha_cumprod_t_next = torch.sqrt(alpha_cumprod_t_next)
#         sqrt_one_minus_alpha_cumprod_t_next = torch.sqrt(1 - alpha_cumprod_t_next)
        
#         # Direction pointing to x_t
#         dir_xt = sqrt_one_minus_alpha_cumprod_t_next * predicted_noise
        
#         # Deterministic part
#         x_next = sqrt_alpha_cumprod_t_next * pred_x0 + dir_xt
        
#         # Add stochastic part if eta > 0
#         if eta > 0 and t_next > 0:
#             # Compute variance
#             alpha_t = alpha_cumprod_t / alpha_cumprod_t_next if t_next >= 0 else alpha_cumprod_t
#             beta_t = 1 - alpha_t
#             sigma_t = eta * torch.sqrt(beta_t * (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t))
            
#             # Add noise
#             noise = torch.randn_like(x)
#             x_next = x_next + sigma_t * noise
            
#         return x_next

#     def make_ddim_timesteps(self, ddim_num_steps, ddim_discr_method="uniform", ddim_eta=0.0):
#         """
#         Create a subset of timesteps to use for DDIM sampling.
        
#         Args:
#             ddim_num_steps: number of steps for DDIM sampling (e.g., 50 instead of 1000)
#             ddim_discr_method: how to choose timesteps ("uniform" or "quad")
#             ddim_eta: stochasticity parameter (0.0 = deterministic)
#         """
#         if ddim_discr_method == 'uniform':
#             c = self.num_timesteps // ddim_num_steps
#             ddim_timesteps = np.asarray(list(range(0, self.num_timesteps, c)))
#         elif ddim_discr_method == 'quad':
#             ddim_timesteps = ((np.linspace(0, np.sqrt(self.num_timesteps * .8), ddim_num_steps)) ** 2).astype(int)
#         else:
#             raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

#         # Add one to get the final alpha cumprod
#         steps_out = ddim_timesteps + 1
#         return steps_out

#     @torch.no_grad()
#     def ddim_sample_loop(self, x_in, ddim_num_steps=50, ddim_eta=0.0, continous=False, sample_inter=None):
#         """
#         DDIM sampling loop - much faster than DDPM with fewer steps.
        
#         Args:
#             x_in: input low-resolution image
#             ddim_num_steps: number of sampling steps (default: 50, much less than 1000 for DDPM)
#             ddim_eta: stochasticity (0.0 = deterministic, 1.0 = stochastic like DDPM)
#             continous: whether to return intermediate steps
#             sample_inter: interval for saving intermediate results
#         """
#         _, _, w, h = x_in.shape

#         # print("ddim shape: ", x_in.shape)
#         x_in = self.apply_dwt(x_in)
#         # print("ddim shape after dwt: ", x_in.shape)
#         device = self.betas.device
        
#         # Create timestep schedule
#         timesteps = self.make_ddim_timesteps(ddim_num_steps, ddim_discr_method="uniform", ddim_eta=ddim_eta)
#         timesteps = timesteps[::-1]  # Reverse for sampling
        
#         if sample_inter is None:
#             sample_inter = max(1, ddim_num_steps // 10)
        
#         x_sr = self.dwsr(x_in)
#         # print("ddim after dwsr: ", x_sr.shape)
        
#         if not self.conditional:
#             shape = x_in.shape
#             img = torch.randn(shape, device=device)
#             ret_img = img.clone()
            
#             for i, (t, t_next) in enumerate(tqdm(zip(timesteps[:-1], timesteps[1:]), 
#                                                 desc=f"DDIM sampling ({ddim_num_steps} steps)", 
#                                                 total=len(timesteps)-1)):
#                 img = self.ddim_step(img, t-1, t_next-1, eta=ddim_eta)
#                 if i % sample_inter == 0:
#                     ret_img = torch.cat([ret_img, img], dim=0)
            
#             # Final step to t=0
#             img = self.ddim_step(img, timesteps[-1]-1, -1, eta=ddim_eta)
#             ret_img = torch.cat([ret_img, img], dim=0)
            
#         else:
#             x = x_in
#             shape = x.shape
#             img = torch.randn(shape, device=device)
#             ret_img = x.clone()
            
#             for i, (t, t_next) in enumerate(tqdm(zip(timesteps[:-1], timesteps[1:]), 
#                                                 desc=f"DDIM sampling ({ddim_num_steps} steps)", 
#                                                 total=len(timesteps)-1)):
#                 img = self.ddim_step(img, t-1, t_next-1, condition_x=x, eta=ddim_eta)
#                 if i % sample_inter == 0:
#                     ret_img = torch.cat([ret_img, img], dim=0)
            
#             # Final step to t=0  
#             img = self.ddim_step(img, timesteps[-1]-1, -1, condition_x=x, eta=ddim_eta)
#             ret_img = torch.cat([ret_img, img], dim=0)

#         if continous:
#             result = self.apply_idwt((ret_img[0]).unsqueeze(0), w, h)
#             result = torch.cat([result, self.apply_idwt((ret_img[0] + x_sr[0]).unsqueeze(0), w, h)], 0)
#             for i in range(1, len(ret_img)):
#                 result = torch.cat([result, self.apply_idwt((ret_img[i] + x_sr[0]).unsqueeze(0), w, h)], 0)
#             return result
#         else:
#             # print("SR: ", x_sr.shape, "Ret_img: ", ret_img[-24:].shape)
#             result = self.apply_idwt((ret_img[-x_sr.shape[0]:] + x_sr), w, h)
#             return result

#     @torch.no_grad()
#     def sample(self, batch_size=1, continous=False, sample_inter=None):
#         image_size = self.image_size
#         channels = self.channels
#         return self.p_sample_loop((batch_size, channels, image_size, image_size), continous, sample_inter)

#     @torch.no_grad()
#     def super_resolution(self, x_in, continous=False, sample_inter=None, use_ddim=True, ddim_steps=50, ddim_eta=0.0):
#         """
#         Super resolution with option to use DDIM or DDPM sampling.
        
#         Args:
#             x_in: input low-resolution image
#             continous: whether to return intermediate results
#             sample_inter: interval for intermediate results
#             use_ddim: whether to use DDIM (True) or DDPM (False) sampling
#             ddim_steps: number of steps for DDIM (ignored if use_ddim=False)
#             ddim_eta: stochasticity for DDIM (0.0 = deterministic)
#         """
#         logger.debug(f"Super resolution for {x_in.shape} with continous={continous}")
        
#         if use_ddim:
#             return self.ddim_sample_loop(x_in, ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, 
#                                        continous=continous, sample_inter=sample_inter)
#         else:
#             return self.p_sample_loop(x_in, continous, sample_inter)

#     # ===== DDIM SAMPLE METHOD FOR COMPATIBILITY =====
#     @torch.no_grad()
#     def ddim_sample(self, batch_size=1, ddim_steps=50, ddim_eta=0.0, continous=False, sample_inter=None):
#         """DDIM sampling method for unconditional generation"""
#         image_size = self.image_size
#         channels = self.channels
#         dummy_input = torch.zeros(batch_size, channels, image_size, image_size)
#         return self.ddim_sample_loop(dummy_input, ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, 
#                                    continous=continous, sample_inter=sample_inter)

#     # ===== TRAINING METHODS (UNCHANGED) =====
#     def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
#         noise = default(noise, lambda: torch.randn_like(x_start))
#         return continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise

#     def p_losses(self, x_in, noise=None):
#         x_start = self.apply_dwt(x_in["HR"])
#         [b, c, h, w] = x_start.shape
#         t = np.random.randint(1, self.num_timesteps + 1)
#         continuous_sqrt_alpha_cumprod = torch.FloatTensor(
#             np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b)
#         ).to(x_start.device)
#         continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

#         noise = default(noise, lambda: torch.randn_like(x_start))
#         x_lr = self.apply_dwt(x_in["SR"])
#         x_sr = self.dwsr(x_lr)
#         x_noisy = self.q_sample(
#             x_start=x_start - x_sr,
#             continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
#             noise=noise,
#         )

#         if not self.conditional:
#             x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
#         else:
#             x_recon = self.denoise_fn(torch.cat([x_lr, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

#         loss = self.loss_func(noise, x_recon)
#         return loss

#     def forward(self, x, *args, **kwargs):
#         return self.p_losses(x, *args, **kwargs)

import logging
import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "quad":
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DWSR(nn.Module):
    def __init__(self, in_channels, features, out_channels, kernel_size=3, padding=1, stride=1, groups=1, depth=10):
        super(DWSR, self).__init__()

        self.conv_layers = [
            nn.Conv2d(
                in_channels,
                features,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                groups=groups,
            ),
            nn.ReLU(),
        ]
        for i in range(depth):
            self.conv_layers.append(
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=groups
                )
            )
            self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(
            nn.Conv2d(
                features,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                groups=groups
            )
        )
        self.convs = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        return self.convs(x) + x


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, image_size, channels=4, loss_type="l1", conditional=True, schedule_opt=None):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.dwsr = DWSR(16, 64, 16, depth=10)
        self.loss_type = loss_type
        self.conditional = conditional
        self.xfm = DWTForward(J=1, mode="zero", wave="haar")
        self.ifm = DWTInverse(mode="zero", wave="haar")
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

        print("Channels: ", self.channels)

    def set_loss(self, device):
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="sum").to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="sum").to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt["schedule"],
            n_timestep=schedule_opt["n_timestep"],
            linear_start=schedule_opt["linear_start"],
            linear_end=schedule_opt["linear_end"],
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer(
            "posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        )

    def apply_dwt(self, images):
        images_LL, hfreq_tuple = self.xfm(images)

        return torch.cat(
            [images_LL, hfreq_tuple[0][:, :, 0, :, :], hfreq_tuple[0][:, :, 1, :, :], hfreq_tuple[0][:, :, 2, :, :]], 1
        )

    def apply_idwt(self, features, targetW, targetH):
        sr_images_LL = features[:, 0:4, :, :]#changed it from 0:3
        sr_images_HL = features[:, 4:8, :, :].unsqueeze(2)
        sr_images_LH = features[:, 8:12, :, :].unsqueeze(2)
        sr_images_HH = features[:, 12:16, :, :].unsqueeze(2)

        sr_HFreqs = torch.cat([sr_images_HL, sr_images_LH, sr_images_HH], 2)
        sr_images = self.ifm((sr_images_LL, [sr_HFreqs]))
        sr_images = F.interpolate(sr_images, size=(targetW, targetH), mode="bicubic")

        return sr_images

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
            )
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    # ===== ORIGINAL DDPM SAMPLING =====
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, sample_inter=None):
        """Original DDPM sampling loop - keeps all timesteps"""
        _, _, w, h = x_in.shape
        x_in = self.apply_dwt(x_in)
        device = self.betas.device
        if sample_inter is None:
            sample_inter = 1 | (self.num_timesteps // 10)
        x_sr = self.dwsr(x_in)
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(
                reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps
            ):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(
                reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps
            ):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            result = self.apply_idwt((ret_img[0]).unsqueeze(0), w, h)
            result = torch.cat([result, self.apply_idwt((ret_img[0] + x_sr[0]).unsqueeze(0), w, h)], 0)
            for i in range(1, len(ret_img)):
                result = torch.cat([result, self.apply_idwt((ret_img[i] + x_sr[0]).unsqueeze(0), w, h)], 0)
            return result
        else:
            # print("SR: ", x_sr.shape, "Ret_img: ", ret_img[-24:].shape)
            result = self.apply_idwt((ret_img[-x_sr.shape[0]:] + x_sr), w, h)
            return result

    # ===== NEW DDIM SAMPLING METHODS =====
    @torch.no_grad()
    def ddim_step(self, x, t, t_next, clip_denoised=True, condition_x=None, eta=0.0):
        """
        Perform one DDIM step from timestep t to timestep t_next.
        
        Args:
            x: current noisy image
            t: current timestep (0-based index)
            t_next: next timestep (should be < t, -1 for final step)
            clip_denoised: whether to clip predicted x_0
            condition_x: conditioning input (for conditional generation)
            eta: amount of stochasticity (0 = deterministic DDIM, 1 = DDPM)
        """
        batch_size = x.shape[0]
        
        # Handle noise level calculation to match your original implementation
        if t < len(self.sqrt_alphas_cumprod_prev) - 1:
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        else:
            # For the highest timestep, use the last available value
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[-1]]).repeat(batch_size, 1).to(x.device)
        
        # Predict noise using the denoising network
        if condition_x is not None:
            predicted_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            predicted_noise = self.denoise_fn(x, noise_level)
        
        # Get alpha values for current and next timesteps
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0).to(x.device)
        
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        # Predict original sample (x_0) - this should match your predict_start_from_noise method
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        
        # DDIM sampling equation
        sqrt_alpha_cumprod_t_next = torch.sqrt(alpha_cumprod_t_next)
        sqrt_one_minus_alpha_cumprod_t_next = torch.sqrt(1 - alpha_cumprod_t_next)
        
        # Deterministic direction
        dir_xt = sqrt_one_minus_alpha_cumprod_t_next * predicted_noise
        
        # Compute x_{t_next}
        x_next = sqrt_alpha_cumprod_t_next * pred_x0 + dir_xt
        
        # Add stochastic component if eta > 0 (makes it more like DDPM)
        if eta > 0 and t_next >= 0:
            sigma = eta * torch.sqrt((1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_next)
            noise = torch.randn_like(x)
            x_next = x_next + sigma * noise
            
        return x_next

    def make_ddim_timesteps(self, ddim_num_steps, ddim_discr_method="uniform"):
        """
        Create a subset of timesteps to use for DDIM sampling.
        
        Args:
            ddim_num_steps: number of steps for DDIM sampling (e.g., 50 instead of 1000)
            ddim_discr_method: how to choose timesteps ("uniform" or "quad")
        """
        if ddim_discr_method == 'uniform':
            # Create uniform spacing, ensuring we include the last timestep
            step_ratio = self.num_timesteps // ddim_num_steps
            ddim_timesteps = np.arange(0, self.num_timesteps, step_ratio)
            # Make sure we don't exceed num_timesteps-1
            ddim_timesteps = ddim_timesteps[ddim_timesteps < self.num_timesteps]
            # Ensure we end at the highest timestep
            if ddim_timesteps[-1] != self.num_timesteps - 1:
                ddim_timesteps = np.append(ddim_timesteps, self.num_timesteps - 1)
        elif ddim_discr_method == 'quad':
            ddim_timesteps = ((np.linspace(0, np.sqrt(self.num_timesteps * .8), ddim_num_steps)) ** 2).astype(int)
            ddim_timesteps = np.clip(ddim_timesteps, 0, self.num_timesteps - 1)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

        return ddim_timesteps

    @torch.no_grad()
    def ddim_sample_loop(self, x_in, ddim_num_steps=50, ddim_eta=0.0, continous=False, sample_inter=None):
        """
        DDIM sampling loop - much faster than DDPM with fewer steps.
        
        Args:
            x_in: input low-resolution image
            ddim_num_steps: number of sampling steps (default: 50, much less than 1000 for DDPM)
            ddim_eta: stochasticity (0.0 = deterministic, 1.0 = stochastic like DDPM)
            continous: whether to return intermediate steps
            sample_inter: interval for saving intermediate results
        """
        _, _, w, h = x_in.shape
        x_in = self.apply_dwt(x_in)
        device = self.betas.device
        
        # Create timestep schedule - start from highest noise and go down
        timesteps = self.make_ddim_timesteps(ddim_num_steps, ddim_discr_method="uniform")
        # Reverse for sampling (start from highest timestep)
        timesteps = timesteps[::-1]  
        
        if sample_inter is None:
            sample_inter = max(1, len(timesteps) // 10)
        
        x_sr = self.dwsr(x_in)
        
        if not self.conditional:
            shape = x_in.shape
            img = torch.randn(shape, device=device)
            ret_img = img.clone()
            
            # Sample through the timesteps
            for i in range(len(timesteps)):
                t = timesteps[i]
                t_next = timesteps[i + 1] if i < len(timesteps) - 1 else -1
                
                img = self.ddim_step(img, t, t_next, eta=ddim_eta)
                
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            
            # Make sure we include the final result
            if (len(timesteps) - 1) % sample_inter != 0:
                ret_img = torch.cat([ret_img, img], dim=0)
            
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x.clone()
            
            # Sample through the timesteps
            for i in range(len(timesteps)):
                t = timesteps[i]
                t_next = timesteps[i + 1] if i < len(timesteps) - 1 else -1
                
                img = self.ddim_step(img, t, t_next, condition_x=x, eta=ddim_eta)
                
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            
            # Make sure we include the final result
            if (len(timesteps) - 1) % sample_inter != 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            result = self.apply_idwt((ret_img[0]).unsqueeze(0), w, h)
            result = torch.cat([result, self.apply_idwt((ret_img[0] + x_sr[0]).unsqueeze(0), w, h)], 0)
            for i in range(1, len(ret_img)):
                result = torch.cat([result, self.apply_idwt((ret_img[i] + x_sr[0]).unsqueeze(0), w, h)], 0)
            return result
        else:
            print("SR: ", x_sr.shape, "Ret_img: ", ret_img[-24:].shape)
            result = self.apply_idwt((ret_img[-x_sr.shape[0]:] + x_sr), w, h)
            return result

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False, sample_inter=None):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous, sample_inter)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, sample_inter=None, use_ddim=True, ddim_steps=50, ddim_eta=0.0):
        """
        Super resolution with option to use DDIM or DDPM sampling.
        
        Args:
            x_in: input low-resolution image
            continous: whether to return intermediate results
            sample_inter: interval for intermediate results
            use_ddim: whether to use DDIM (True) or DDPM (False) sampling
            ddim_steps: number of steps for DDIM (ignored if use_ddim=False)
            ddim_eta: stochasticity for DDIM (0.0 = deterministic)
        """
        logger.debug(f"Super resolution for {x_in.shape} with continous={continous}")
        
        if use_ddim:
            return self.ddim_sample_loop(x_in, ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, 
                                       continous=continous, sample_inter=sample_inter)
        else:
            return self.p_sample_loop(x_in, continous, sample_inter)

    # ===== DDIM SAMPLE METHOD FOR COMPATIBILITY =====
    @torch.no_grad()
    def ddim_sample(self, batch_size=1, ddim_steps=50, ddim_eta=0.0, continous=False, sample_inter=None):
        """DDIM sampling method for unconditional generation"""
        image_size = self.image_size
        channels = self.channels
        dummy_input = torch.zeros(batch_size, channels, image_size, image_size)
        return self.ddim_sample_loop(dummy_input, ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, 
                                   continous=continous, sample_inter=sample_inter)

    # ===== TRAINING METHODS (UNCHANGED) =====
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise

    def p_losses(self, x_in, noise=None):
        x_start = self.apply_dwt(x_in["HR"])
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_lr = self.apply_dwt(x_in["SR"])
        x_sr = self.dwsr(x_lr)
        x_noisy = self.q_sample(
            x_start=x_start - x_sr,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=noise,
        )

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(torch.cat([x_lr, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)