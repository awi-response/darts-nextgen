import copy
import math
import os
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Type, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
from diffusers.utils.torch_utils import randn_tensor
from pytorch_lightning import LightningModule
from torch import nn, optim, Tensor
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
# import vutils
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
import tifffile
from darts_superresolution.model.wave_modules import diffusion, unet
from darts_superresolution.config.configuration_inference import InferenceConfig as Config
from darts_superresolution.core import metrics as Metrics

class DWSR(nn.Module):
    """Deep Wavelet Super-Resolution network - copied from your diffusion model"""
    def __init__(self, in_channels, features, out_channels, kernel_size=3, padding=1, stride=1, groups=1, depth=10):
        super(DWSR, self).__init__()

        self.conv_layers = [
            nn.Conv2d(in_channels, 
                      features, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      stride=stride, 
                      bias=False, 
                      groups=groups),
            nn.ReLU()
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


class ConsistencyWavelet(LightningModule):
    """ Consistency model with wavelet transform integration """

    def __init__(
        self,
        config: Config,
        bins_min: int = 5,
        bins_max: int = 150,
        bins_rho: float = 7,
        loss_func: str = 'HybridWavelet',#'LPIPS',
        initial_ema_decay: float = 0.90,
        optimizer_type: Type[optim.Optimizer] = optim.RAdam,
        num_samples: int = 16,
        use_ema: bool = True,
        sample_seed: int = 0,
        log_images_every_n_steps: int = 200,
        log_loss_every_n_steps: int = 100,
        diffusion_checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> None:

        super().__init__()
        
        self.save_hyperparameters(ignore=['loss_fn'])
        self.config = config()

        # Initialize wavelet transforms (same as diffusion model)
        self.xfm = DWTForward(J=1, mode='zero', wave='haar')
        self.ifm = DWTInverse(mode='zero', wave='haar')
        
        # Initialize DWSR network (same as diffusion model)
        self.dwsr = DWSR(16, 64, 16, depth=10)  # Match your diffusion model

        # Initialize UNet with same architecture as diffusion model
        if ('norm_groups' not in self.config.unet) or self.config.unet['norm_groups'] is None:
            self.config.unet['norm_groups'] = 32
        
        model = unet.UNet(
            in_channel=self.config.unet['in_channel'],
            out_channel=self.config.unet['out_channel'],
            norm_groups=self.config.unet['norm_groups'],
            inner_channel=self.config.unet['inner_channel'],
            channel_mults=self.config.unet['channel_multiplier'],
            attn_res=self.config.unet['attn_res'],
            res_blocks=self.config.unet['res_blocks'],
            dropout=self.config.unet['dropout'],
            image_size=self.config.unet['diffusion']['image_size']
        )

        self.model = model
        

        # Load from diffusion checkpoint if provided
        if diffusion_checkpoint_path and os.path.exists(diffusion_checkpoint_path):  # Added check for None/empty
            self.load_from_diffusion_checkpoint(diffusion_checkpoint_path)
            print(f"Loaded diffusion model weights from: {diffusion_checkpoint_path}")

        # # Add noise schedule parameters (matching your diffusion model)
        # self.register_buffer('sqrt_alphas_cumprod_prev', torch.tensor([1.0]))  # Placeholder
        # if diffusion_checkpoint_path and os.path.exists(diffusion_checkpoint_path):
        #     self.load_noise_schedule_from_checkpoint(diffusion_checkpoint_path)

        # Add noise schedule parameters (matching your diffusion model)
        self.register_buffer('sqrt_alphas_cumprod_prev', torch.tensor([1.0]))  # Placeholder
        if diffusion_checkpoint_path and os.path.exists(diffusion_checkpoint_path):  # Added check
            self.load_noise_schedule_from_checkpoint(diffusion_checkpoint_path)
        
        # Create EMA model after loading weights
        self.model_ema = copy.deepcopy(self.model)
        self.model_ema.requires_grad_(False)
        
        # Copy DWSR to EMA as well
        self.dwsr_ema = copy.deepcopy(self.dwsr)
        self.dwsr_ema.requires_grad_(False)
        
        self.image_size = self.config.sample_dimension

        # # Loss function
        # if loss_func == "LPIPS":
        #     self.loss_fn = PerceptualLoss(net_type="squeeze")
        # elif loss_func == "MSE":
        #     self.loss_fn = nn.MSELoss()
        # elif loss_func == "L1":
        #     self.loss_fn = nn.L1Loss()
        # elif loss_func == "WeightedWavelet":
        #     # self.loss_fn = WeightedWaveletLoss()
        #     self.wavelet_loss_fn = WeightedWaveletLoss()
        #     self.loss_fn = nn.MSELoss()  # Fallback
        # elif loss_func == "AdaptiveWavelet": 
        #     # self.loss_fn = AdaptiveWaveletLoss()
        #     self.wavelet_loss_fn = AdaptiveWaveletLoss()
        #     self.loss_fn = nn.MSELoss()  # Fallback
        # elif loss_func == "HybridWavelet":
        #     # self.loss_fn = HybridWaveletImageLoss(xfm=self.xfm, ifm=self.ifm)
        #     self.wavelet_loss_fn = HybridWaveletImageLoss(
        #         wavelet_weight=0.2,      # Low weight on wavelets
        #         image_weight=0.8,         # High weight on image quality
        #         perceptual_weight=1.0,    # Perceptual in image domain
        #         l1_weight=0.5,
        #         xfm=self.xfm,
        #         ifm=self.ifm,
        #         )
        #     self.loss_fn = nn.MSELoss()
        # else:
        #     print("loss function not defined.")
        # print("Using: ", loss_func)

        # if self.config.use_regularization:
        #     self.tv_loss = TotalVariationLoss()


        # ## Metrics
        # self.lpips_model = None  # Set to 'alexnet' or 'vgg' if you want LPIPS
        # self.clip_model = 'clip-ViT-B/16'  # Set to 'clip-ViT-B/16' if you want CLIP score
        
        # # Trackers for metrics
        # self._val_psnr_tracker = MeanMetric()
        # self._val_ssim_tracker = MeanMetric()

        # self.optimizer_type = optimizer_type
        # self.learning_rate = self.config.lr
        # self.initial_ema_decay = initial_ema_decay

        # self.data_std = self.config.data_std
        # self.time_min = self.config.time_min 
        # self.time_max = self.config.time_max
        # self.clip = self.config.clip_output

        # self.bins_min = bins_min
        # self.bins_max = bins_max
        # self.bins_rho = bins_rho

        # self._train_loss_tracker = MeanMetric()
        # self._val_loss_tracker = MeanMetric()
        # self._bins_tracker = MeanMetric()
        # self._ema_decay_tracker = MeanMetric()

        # self.num_samples = num_samples
        # self.use_ema = use_ema
        # self.sample_seed = sample_seed
        # self.sample_steps = 10 #instead of 1
        # self.log_images_every_n_steps = log_images_every_n_steps
        # self.log_loss_every_n_steps = log_loss_every_n_steps

        # self.is_conditional_model = True

        # Inference/runtime attributes used by sample_conditional and helpers.
        # These were commented during the port but are required at inference time.
        self.optimizer_type = optimizer_type
        self.learning_rate = getattr(self.config, "lr", 1e-4)
        self.initial_ema_decay = initial_ema_decay

        self.data_std = float(getattr(self.config, "data_std", 0.5))
        self.time_min = float(getattr(self.config, "time_min", 0.002))
        self.time_max = float(getattr(self.config, "time_max", 80.0))
        self.clip = bool(getattr(self.config, "clip_output", False))

        self.bins_min = bins_min
        self.bins_max = bins_max
        self.bins_rho = bins_rho

        self.num_samples = num_samples
        self.use_ema = use_ema
        self.sample_seed = sample_seed
        self.sample_steps = 10
        self.log_images_every_n_steps = log_images_every_n_steps
        self.log_loss_every_n_steps = log_loss_every_n_steps

        self.is_conditional_model = True

    def apply_dwt(self, images):
        """Apply DWT transform - same as diffusion model"""
        images_LL, hfreq_tuple = self.xfm(images)
        return torch.cat([images_LL,
                          hfreq_tuple[0][:, :, 0, :, :],
                          hfreq_tuple[0][:, :, 1, :, :],
                          hfreq_tuple[0][:, :, 2, :, :]], 1)

    def apply_idwt(self, features, target_w, target_h):
        """Apply inverse DWT transform - same as diffusion model"""
        sr_images_LL = features[:, 0:4, :, :]
        sr_images_HL = features[:, 4:8, :, :].unsqueeze(2)
        sr_images_LH = features[:, 8:12, :, :].unsqueeze(2)
        sr_images_HH = features[:, 12:16, :, :].unsqueeze(2)

        sr_HFreqs = torch.cat([sr_images_HL, sr_images_LH, sr_images_HH], 2)
        sr_images = self.ifm((sr_images_LL, [sr_HFreqs]))
        sr_images = F.interpolate(sr_images, size=(target_w, target_h), mode='bicubic')
        return sr_images

    def load_from_diffusion_checkpoint(self, checkpoint_path: str):
        """Load weights from a diffusion model checkpoint including DWSR"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                
                # Separate UNet and DWSR parameters
                unet_state_dict = {}
                dwsr_state_dict = {}
                
                for key, value in state_dict.items():
                    if 'denoise_fn' in key:
                        # Remove 'denoise_fn.' prefix for UNet parameters
                        new_key = key.replace('denoise_fn.', '')
                        unet_state_dict[new_key] = value
                    elif 'dwsr' in key:
                        # Remove any module prefix for DWSR parameters
                        new_key = key.replace('dwsr.', '').replace('module.dwsr.', '')
                        dwsr_state_dict[new_key] = value
                
                # Load UNet weights
                missing_keys, unexpected_keys = self.model.load_state_dict(unet_state_dict, strict=False)
                if missing_keys:
                    print(f"Missing UNet keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected UNet keys: {unexpected_keys}")
                
                # Load DWSR weights
                missing_keys, unexpected_keys = self.dwsr.load_state_dict(dwsr_state_dict, strict=False)
                if missing_keys:
                    print(f"Missing DWSR keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected DWSR keys: {unexpected_keys}")

                # Test DWSR initialization
                # self.test_dwsr_initialization()
                    
        except Exception as e:
            print(f"Error loading diffusion checkpoint: {e}")
            print("Proceeding with random initialization...")
            self.initialize_dwsr_for_wavelets()

    def initialize_dwsr_for_wavelets(self):
        """Proper initialization for DWSR in wavelet domain"""
        print("Initializing DWSR for wavelet domain...")
        for module in self.dwsr.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier initialization works better for wavelet coefficients
                nn.init.xavier_normal_(module.weight, gain=0.1)  # Small gain for residual learning


    def configure_optimizers(self):
        # Include both UNet and DWSR parameters in optimization
        params = list(self.model.parameters()) + list(self.dwsr.parameters())
        optimizer = self.optimizer_type(params, lr=self.learning_rate)

        # Make T_max based on total training steps, not a fixed number
        total_steps = self.trainer.estimated_stepping_batches
    
        # Add learning rate scheduler for consistency models
        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=1e-6
            ),
            'interval': 'step',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, images: torch.Tensor, times: torch.Tensor, conditioning=None):
        return self._forward_conditional(self.model, self.dwsr, images, times, conditioning)

    def _forward_conditional(self, model, dwsr_model, images, times, conditioning=None):
        """Forward pass with wavelet transforms and conditioning support - FIXED VERSION"""
    
        # images and conditioning should already be in wavelet domain when called
        # No need to apply DWT here since inputs are already transformed
        
        if conditioning is not None:
            # Both inputs should already be in wavelet domain
            conditioning_dwt = conditioning  # Already in wavelet domain
            images_dwt = images  # Already in wavelet domain
            
            # Process conditioning through DWSR to get baseline
            conditioning_sr = dwsr_model(conditioning_dwt)
            
            # For consistency model, we predict the residual directly
            # Input to UNet: [conditioning, target_residual] where target_residual = target - baseline
            target_residual = images_dwt - conditioning_sr
            model_input = torch.cat([conditioning_dwt, target_residual], dim=1)
        else:
            model_input = images  # Should already be in wavelet domain
        
        # Ensure times has correct dtype and shape for the UNet
        if times.dim() == 1:
            times = times.unsqueeze(-1)
        times = times.float()
        
        # Calculate coefficients for consistency model
        skip_coef = self.data_std**2 / ((times.squeeze(-1) - self.time_min).pow(2) + self.data_std**2)
        out_coef = self.data_std * times.squeeze(-1) / (times.squeeze(-1).pow(2) + self.data_std**2).pow(0.5)
        
        # Convert consistency times to diffusion-style noise levels
        noise_levels = self.times_to_noise_levels(times)
        predicted_residual = model(model_input, noise_levels)
        
        # Apply consistency model skip connections in wavelet domain
        if conditioning is not None:
            # Output = skip_connection(target_residual) + predicted_residual + sr_baseline
            # This gives us the final wavelet representation
            out_dwt = (self.image_time_product(target_residual, skip_coef) + 
                    self.image_time_product(predicted_residual, out_coef) + 
                    conditioning_sr)
        else:
            # Unconditional case
            out_dwt = (self.image_time_product(images, skip_coef) + 
                    self.image_time_product(predicted_residual, out_coef))
        
        if self.clip:
            out_dwt = out_dwt.clamp(-1.0, 1.0)
        
        return out_dwt

    # def _forward_conditional_noise_prediction(self, model, dwsr_model, noisy_residual, times, conditioning_dwt, x_sr):
    #     """Forward pass that predicts noise in the residual (for training)"""
        
    #     # Input to UNet: [conditioning, noisy_residual]
    #     model_input = torch.cat([conditioning_dwt, noisy_residual], dim=1)
        
    #     # Ensure times has correct format
    #     if times.dim() == 1:
    #         times = times.unsqueeze(-1)
    #     times = times.float()
        
    #     # Convert to noise levels for UNet
    #     noise_levels = self.times_to_noise_levels(times)
        
    #     # Model predicts the noise component
    #     predicted_noise = model(model_input, noise_levels)
        
    #     return predicted_noise


    def training_step(self, batch, *args, **kwargs):
        """Training step - FIXED to work properly in wavelet domain"""
        lr_images = batch['lr'] 
        hr_images = batch['hr'] 
        
        # Convert to wavelet domain immediately
        lr_images_dwt = self.apply_dwt(lr_images)
        hr_images_dwt = self.apply_dwt(hr_images)
        
        # Get DWSR baseline in wavelet domain
        # x_sr = self.dwsr(lr_images_dwt)
        
        # Target is the clean residual in wavelet domain
        # x_clean_residual = hr_images_dwt - x_sr
        
        # residual_std = torch.std(x_clean_residual)
        # if residual_std > 3.0:  # If residuals are too large
            # x_clean_residual = x_clean_residual / residual_std * 2.0
    
        # In training_step, add after getting x_sr:
        if self.trainer.global_step < 10:  # Log first few steps
            print(f"Step {self.trainer.global_step}:")
            # print(f"DWSR output range: [{torch.min(x_sr):.3f}, {torch.max(x_sr):.3f}]")
            print(f"Target range: [{torch.min(hr_images_dwt):.3f}, {torch.max(hr_images_dwt):.3f}]")
            # print(f"Residual range: [{torch.min(x_clean_residual):.3f}, {torch.max(x_clean_residual):.3f}]")
        
        _bins = self.bins
        
        # Generate noise matching residual shape
        # noise = torch.randn_like(x_clean_residual)
        noise = torch.randn_like(hr_images_dwt)
        
        timesteps = torch.randint(0, _bins - 1, (hr_images.shape[0],), device=hr_images.device).long()
        
        current_times = self.timesteps_to_times(timesteps, _bins)
        next_times = self.timesteps_to_times(timesteps + 1, _bins)
        
        # Add noise to clean residual
        # current_noisy_residual = x_clean_residual + self.image_time_product(noise, current_times.squeeze(-1))
        # next_noisy_residual = x_clean_residual + self.image_time_product(noise, next_times.squeeze(-1))
        
        current_noisy = hr_images_dwt + self.image_time_product(noise, current_times.squeeze(-1))
        next_noisy = hr_images_dwt + self.image_time_product(noise, next_times.squeeze(-1))
    

        # Get target noise prediction from EMA model
        # with torch.no_grad():
        #     target_noise = self._forward_conditional_noise_prediction(
        #         self.model_ema, self.dwsr_ema, current_noisy_residual, current_times, lr_images_dwt, x_sr
        #     )
        
        # # Get current model's noise prediction
        # predicted_noise = self._forward_conditional_noise_prediction(
        #     self.model, self.dwsr, next_noisy_residual, next_times, lr_images_dwt, x_sr
        # )
        with torch.no_grad():
            target_output = self._forward_conditional(
            self.model_ema, self.dwsr_ema, current_noisy, current_times, lr_images_dwt
        )
    
        predicted_output = self._forward_conditional(
            self.model, self.dwsr, next_noisy, next_times, lr_images_dwt
        )
        
        # Loss compares noise predictions in wavelet domain
        if hasattr(self, 'wavelet_loss_fn'):
            if type(self.wavelet_loss_fn) == HybridWaveletImageLoss:
                _, _, h, w = hr_images.shape
                loss = self.wavelet_loss_fn(predicted_output, target_output, h, w)
            else:
                loss = self.wavelet_loss_fn(predicted_output, target_output)

            if hasattr(self.wavelet_loss_fn, 'get_sub_losses'):
                sub_losses = self.wavelet_loss_fn.get_sub_losses()
                for loss_name, loss_value in sub_losses.items():
                    self.log(
                        f"train_loss/{loss_name}", 
                        loss_value, 
                        on_step=False, 
                        on_epoch=True, 
                        logger=True
                    )
        else:
            loss = self.loss_fn(predicted_output, target_output)
            if hasattr(self.loss_fn, 'get_sub_losses'):
                sub_losses = self.loss_fn.get_sub_losses()
                for loss_name, loss_value in sub_losses.items():
                    self.log(
                        f"train_loss/{loss_name}", 
                        loss_value, 
                        on_step=False,
                        on_epoch=True, 
                        logger=True
                    )

    
        # Total Variation regularization (no target needed)
        if hasattr(self, 'tv_loss'):
            print("We are using total variation regularization now.")
            pred_image = self.apply_idwt(predicted_output, hr_images.shape[2], hr_images.shape[3])
            tv_penalty = self.tv_loss(pred_image)
            loss = loss + tv_penalty
        
        # loss = self.loss_fn(predicted_noise, target_noise)
        
        # Add spectral regularization to prevent high-frequency artifacts

        ### We can add this regularization if high frequency artifacts occur ###
        # hf_bands = predicted_noise[:, 4:16]  # High frequency bands
        # spectral_penalty = 0.01 * torch.mean(torch.abs(hf_bands))
        # loss = loss + spectral_penalty

        self._train_loss_tracker(loss)

        if self.trainer.global_step % self.log_loss_every_n_steps == 0:
            self.log("train_loss_frequent", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
    
        self.log("train_loss", self._train_loss_tracker, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        

        # Enhanced debugging for first few steps
        if self.trainer.global_step < 20:
            print(f"Step {self.trainer.global_step}:")
            # print(f"  DWSR output range: [{torch.min(x_sr):.3f}, {torch.max(x_sr):.3f}]")
            print(f"  Target range: [{torch.min(hr_images_dwt):.3f}, {torch.max(hr_images_dwt):.3f}]")
            # print(f"  Residual range: [{torch.min(x_clean_residual):.3f}, {torch.max(x_clean_residual):.3f}]")
            # print(f"  Residual std: {torch.std(x_clean_residual):.3f}")
            print(f"  Noise std: {torch.std(noise):.3f}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Times range: [{torch.min(current_times):.3f}, {torch.max(current_times):.3f}]")
            print(f"  Bins: {_bins}")


        # Log samples periodically
        if self.trainer.global_step % self.log_images_every_n_steps == 0:
            self.log_super_resolution_samples(lr_images[:4], hr_images[:4], "train")
        
        return loss


    @torch.no_grad()
    def validation_step(self, batch, *args, **kwargs) -> torch.Tensor:
        """Validation step - FIXED to work properly in wavelet domain"""
        lr_images = batch['lr']
        hr_images = batch['hr']

        _, _, h, w = hr_images.shape
        
        # Convert to wavelet domain immediately
        lr_images_dwt = self.apply_dwt(lr_images)
        hr_images_dwt = self.apply_dwt(hr_images)
        
        # Get DWSR baseline using EMA model
        # x_sr = self.dwsr_ema(lr_images_dwt)
        
        # Target is the clean residual in wavelet domain
        # x_clean_residual = hr_images_dwt - x_sr

        _bins = self.bins
        
        # Generate noise matching residual shape
        # noise = torch.randn_like(x_clean_residual)
        noise = torch.randn_like(hr_images_dwt)
        
        timesteps = torch.randint(0, _bins - 1, (hr_images.shape[0],), device=hr_images.device).long()
        
        current_times = self.timesteps_to_times(timesteps, _bins)
        next_times = self.timesteps_to_times(timesteps + 1, _bins)
        
        # Add noise to clean residual
        # current_noisy_residual = x_clean_residual + self.image_time_product(noise, current_times.squeeze(-1))
        # next_noisy_residual = x_clean_residual + self.image_time_product(noise, next_times.squeeze(-1))
        current_noisy = hr_images_dwt + self.image_time_product(noise, current_times.squeeze(-1))
        next_noisy = hr_images_dwt + self.image_time_product(noise, next_times.squeeze(-1))
        # Get target and predicted noise
        # target_noise = self._forward_conditional_noise_prediction(
        #     self.model_ema, self.dwsr_ema, current_noisy_residual, current_times, lr_images_dwt, x_sr
        # )
        
        # predicted_noise = self._forward_conditional_noise_prediction(
        #     self.model, self.dwsr, next_noisy_residual, next_times, lr_images_dwt, x_sr
        # )


        target_output = self._forward_conditional(
            self.model_ema, self.dwsr_ema, current_noisy, current_times, lr_images_dwt
        )
    
        predicted_output = self._forward_conditional(
            self.model, self.dwsr, next_noisy, next_times, lr_images_dwt
        )
        
        # Loss compares noise predictions in wavelet domain
        # loss = self.loss_fn(predicted_noise, target_noise, h, w)

        if hasattr(self, 'wavelet_loss_fn'):
            # print("Type: ", type(self.loss_fn))
            if type(self.wavelet_loss_fn) == HybridWaveletImageLoss:
                _, _, h, w = hr_images.shape
                loss = self.wavelet_loss_fn(predicted_output, target_output, h, w)
            else:
                loss = self.wavelet_loss_fn(predicted_output, target_output)

                    # ========== LOG SUB-LOSSES ==========
            if hasattr(self.wavelet_loss_fn, 'get_sub_losses'):
                sub_losses = self.wavelet_loss_fn.get_sub_losses()
                for loss_name, loss_value in sub_losses.items():
                    self.log(
                        f"val_loss/{loss_name}", 
                        loss_value, 
                        on_step=False,
                        on_epoch=True,
                        logger=True
                    )
        else:
            loss = self.loss_fn(predicted_output, target_output)
            if hasattr(self.loss_fn, 'get_sub_losses'):
                sub_losses = self.loss_fn.get_sub_losses()
                for loss_name, loss_value in sub_losses.items():
                    self.log(
                        f"val_loss/{loss_name}", 
                        loss_value, 
                        on_step=False,
                        on_epoch=True,
                        logger=True
                    )
        
        self._val_loss_tracker(loss)
        self.log("val_loss", self._val_loss_tracker, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # Log additional metrics
        # if self.trainer.global_step % (self.log_loss_every_n_steps * 5) == 0:
        #     # Log residual statistics
        #     lr_images_dwt = self.apply_dwt(lr_images)
        #     hr_images_dwt = self.apply_dwt(hr_images)
        #     x_sr = self.dwsr_ema(lr_images_dwt)
        #     residual = hr_images_dwt - x_sr
            
            # self.log("val_residual_std", torch.std(residual), on_step=True, logger=True)
            # self.log("val_sr_baseline_std", torch.std(x_sr), on_step=True, logger=True)

            # ========== METRICS CALCULATION ==========
        # Generate actual super-resolved images for metrics
        sr_images, _ = self.sample_conditional(
            conditioning=lr_images,
            x_image_size=h,
            y_image_size=w,
            steps=10,#instead of just 1.
            use_ema=True
        )
        
        # Apply adaptive instance normalization
        # sr_images = self.adaptive_instance_normalization(sr_images, lr_images)
        
        # Convert to numpy for metrics calculation
        # Use tensor2img from your metrics module
        pred_np = Metrics.tensor2img(sr_images.cpu(), out_type=np.uint8, min_max=(-1, 1))
        hr_np = Metrics.tensor2img(hr_images.cpu(), out_type=np.uint8, min_max=(-1, 1))
        
        # Reshape for metrics: need (B, H, W, C)
        if pred_np.ndim == 3:  # Single image (C, H, W) or (H, W, C)
            if pred_np.shape[0] in [1, 3, 4]:  # Likely (C, H, W)
                pred_np = np.transpose(pred_np, (1, 2, 0))
                hr_np = np.transpose(hr_np, (1, 2, 0))
            # Add batch dimension
            pred_np = pred_np[np.newaxis, :, :, :]
            hr_np = hr_np[np.newaxis, :, :, :]
        elif pred_np.ndim == 4:  # Batch (B, C, H, W) or (B, H, W, C)
            if pred_np.shape[1] in [1, 3, 4]:  # Likely (B, C, H, W)
                pred_np = np.transpose(pred_np, (0, 2, 3, 1))
                hr_np = np.transpose(hr_np, (0, 2, 3, 1))
        
        # Calculate PSNR
        try:
            psnr_mean, psnr_batch = Metrics.calculate_psnr(pred_np, hr_np)
            self._val_psnr_tracker(psnr_mean)
            self.log('val_psnr', self._val_psnr_tracker, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            if self.trainer.global_step % 100 == 0:
                print(f"PSNR calculation failed: {e}")
        
        # Calculate SSIM
        try:
            ssim_mean, ssim_batch = Metrics.calculate_ssim(pred_np, hr_np)
            self._val_ssim_tracker(ssim_mean)
            self.log('val_ssim', self._val_ssim_tracker, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            if self.trainer.global_step % 100 == 0:
                print(f"SSIM calculation failed: {e}")
        
        # Optional: LPIPS (slower, enable if needed)
        if self.lpips_model is not None:
            try:
                # Reshape: (B, H, W, C) -> (C, H, W) for single image batch
                if pred_np.shape[0] == 1:
                    pred_lpips = pred_np[0].transpose(2, 0, 1)  # (C, H, W)
                    hr_lpips = hr_np[0].transpose(2, 0, 1)
                else:
                    # For multiple images, process first one or average
                    pred_lpips = pred_np[0].transpose(2, 0, 1)
                    hr_lpips = hr_np[0].transpose(2, 0, 1)
                
                lpips_score = Metrics.calculate_lpips(pred_lpips, hr_lpips, lpips_model=self.lpips_model)
                if lpips_score is not None:
                    self.log('val_lpips', lpips_score, on_step=False, on_epoch=True)
            except Exception as e:
                if self.trainer.global_step % 100 == 0:
                    print(f"LPIPS calculation failed: {e}")
        
        # Optional: CLIP Score (slower, enable if needed)
        if self.clip_model is not None:
            try:
                if pred_np.shape[0] == 1:
                    pred_clip = pred_np[0].transpose(2, 0, 1)
                    hr_clip = hr_np[0].transpose(2, 0, 1)
                else:
                    pred_clip = pred_np[0].transpose(2, 0, 1)
                    hr_clip = hr_np[0].transpose(2, 0, 1)
                
                clip_score = Metrics.calculate_clipscore(pred_clip, hr_clip, clip_model=self.clip_model)
                self.log('val_clip_score', clip_score, on_step=False, on_epoch=True)
            except Exception as e:
                if self.trainer.global_step % 100 == 0:
                    print(f"CLIP score calculation failed: {e}")

        return loss

    ## Would like to keep this simple ##

    # def optimizer_step(self, *args, **kwargs) -> None:
    #     super().optimizer_step(*args, **kwargs)
    #     self.ema_update()

    ### But testing gradient clipping for stability ###
    def optimizer_step(self, *args, **kwargs) -> None:
        """Optimizer step with gradient clipping for stability"""
    
        # Clip gradients before optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.dwsr.parameters(), max_norm=1.0)
    
        super().optimizer_step(*args, **kwargs)
        self.ema_update()

    ### Testing ema update with warmup ###
    @torch.no_grad()
    def ema_update(self):
        """Improved EMA update with warmup"""
        # Warmup period for EMA
        warmup_steps = 1000
        if self.trainer.global_step < warmup_steps:
            # Use higher EMA decay during warmup
            ema_decay = 0.999
        else:
            ema_decay = self.ema_decay
        
        # Update UNet EMA
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]
        torch._foreach_mul_(param_ema, ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - ema_decay)
        
        # Update DWSR EMA
        param_dwsr = [p.data for p in self.dwsr.parameters()]
        param_dwsr_ema = [p.data for p in self.dwsr_ema.parameters()]
        torch._foreach_mul_(param_dwsr_ema, ema_decay)
        torch._foreach_add_(param_dwsr_ema, param_dwsr, alpha=1 - ema_decay)

        self._ema_decay_tracker(ema_decay)
        self.log("ema_decay", self._ema_decay_tracker, on_step=False, on_epoch=True, logger=True)

    # @torch.no_grad()
    # def ema_update(self):
    #     """Update both UNet and DWSR EMA models"""
    #     # Update UNet EMA
    #     param = [p.data for p in self.model.parameters()]
    #     param_ema = [p.data for p in self.model_ema.parameters()]
    #     torch._foreach_mul_(param_ema, self.ema_decay)
    #     torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)
        
    #     # Update DWSR EMA
    #     param_dwsr = [p.data for p in self.dwsr.parameters()]
    #     param_dwsr_ema = [p.data for p in self.dwsr_ema.parameters()]
    #     torch._foreach_mul_(param_dwsr_ema, self.ema_decay)
    #     torch._foreach_add_(param_dwsr_ema, param_dwsr, alpha=1 - self.ema_decay)

    #     self._ema_decay_tracker(self.ema_decay)
    #     self.log("ema_decay", self._ema_decay_tracker, on_step=False, on_epoch=True, logger=True)
    
    @property
    def ema_decay(self):
        return math.exp(self.bins_min * math.log(self.initial_ema_decay) / self.bins)

    @property
    def bins(self, alpha=1.0) -> int:
        # I will try a slow bin growth schedule, to prevent too sudden jumps during early training, so added alpha to scale growth
        current_bins = math.ceil(
            math.sqrt(
                (self.trainer.global_step / self.trainer.estimated_stepping_batches)**alpha * (self.bins_max**2 - self.bins_min**2) + self.bins_min**2
            )
        )
            # Optional: Log to verify correct resumption
        if self.trainer.global_step % 100 == 0:
            print(f"Step {self.trainer.global_step}, Bins: {current_bins}")
        return current_bins
        
        

    def timesteps_to_times(self, timesteps: torch.LongTensor, bins: int):
        """Convert timestep indices to actual time values"""
        # Convert to float first to avoid dtype issues
        timesteps = timesteps.float()
        
        times = (
            (
                self.time_min ** (1 / self.bins_rho)
                + timesteps
                / (bins - 1)
                * (
                    self.time_max ** (1 / self.bins_rho)
                    - self.time_min ** (1 / self.bins_rho)
                )
            )
            .pow(self.bins_rho)
            .clamp(0, self.time_max)
        )
        
        # Ensure it's float32 (not float64) and correct shape
        times = times.float()
        if times.dim() == 1:
            times = times.unsqueeze(-1)
        
        return times


    @torch.no_grad()
    def sample_conditional(
        self,
        conditioning,
        x_image_size,
        y_image_size,
        steps: int = 10,#instead of 1
        sample_times: List = [None],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_ema: bool = False,
    ) -> torch.Tensor:
        """Fast conditional sampling - FIXED to work properly in wavelet domain"""

        conditioning = conditioning.to(self.device)

        # Convert conditioning to wavelet domain
        conditioning_dwt = self.apply_dwt(conditioning)
        
        # Get DWSR baseline
        if use_ema:
            x_sr = self.dwsr_ema(conditioning_dwt)
        else:
            x_sr = self.dwsr(conditioning_dwt)

        # Set up time
        if sample_times[0] is not None:
            time = torch.tensor([sample_times[0]], device=self.device)
        else:
            time = torch.tensor([self.time_max], device=self.device)
        
        if time.dim() == 1:
            time = time.unsqueeze(-1)

        # Match time tensor batch dimension to conditioning batch size.
        # Without this, UNet noise embedding is created for batch=1 and later
        # reshaped as if it matched the image batch (e.g. 24), causing errors.
        if time.shape[0] == 1 and conditioning.shape[0] > 1:
            time = time.expand(conditioning.shape[0], 1)

        # Generate noise for the residual in wavelet domain
        # The residual has same shape as the target wavelet coefficients
        noise_shape = (conditioning.shape[0], 16, conditioning_dwt.shape[2], conditioning_dwt.shape[3])  
        noise = torch.randn(noise_shape, generator=generator, device=self.device)
        
        # Initial noisy residual (start from pure noise)
        noisy_residual = self.image_time_product(noise, time.squeeze(-1))

        model_fn = self.model_ema if use_ema else self.model
        dwsr_fn  = self.dwsr_ema  if use_ema else self.dwsr

        # Initial denoising step from t_max
        full_noisy_target = noisy_residual + x_sr
        output_dwt = self._forward_conditional(
            model_fn, dwsr_fn, full_noisy_target, time, conditioning_dwt
        )

        if steps > 1:
            _timesteps = torch.linspace(self.bins_max - 1, 0, steps, device=self.device).long()
            times_seq  = self.timesteps_to_times(_timesteps, self.bins_max)  # (steps, 1)

            for step_time in times_seq[1:]:  # skip t_max — already done above
                step_time_batch = step_time.expand(conditioning.shape[0], 1)

                noise     = torch.randn_like(output_dwt)
                re_noised = output_dwt + self.image_time_product(
                    noise, step_time_batch.squeeze(-1)
                )

                output_dwt = self._forward_conditional(
                    model_fn, dwsr_fn, re_noised, step_time_batch, conditioning_dwt
                )

        # # Single-step generation in wavelet domain
        # if use_ema:
        #     # Pass the noisy residual and conditioning in wavelet domain
        #     # _forward_conditional expects: images (noisy_residual + x_sr), times, conditioning_dwt
        #     full_noisy_target = noisy_residual + x_sr
        #     output_dwt = self._forward_conditional(
        #         self.model_ema, self.dwsr_ema, full_noisy_target, time, conditioning_dwt
        #     )
        # else:
        #     full_noisy_target = noisy_residual + x_sr
        #     output_dwt = self._forward_conditional(
        #         self.model, self.dwsr, full_noisy_target, time, conditioning_dwt
        #     )

        # Convert back to image domain for output
        _, _, orig_h, orig_w = conditioning.shape
        output_images = self.apply_idwt(output_dwt, orig_h, orig_w)

        return output_images, None


    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b c h w, b -> b c h w", images, times)


    @torch.no_grad()
    def log_super_resolution_samples(self, lr_images, hr_images, stage):
        """Log super-resolution comparison to TensorBoard - FIXED"""
        print("sample conditional shapes: ", lr_images.shape, hr_images.shape)
        
        # Generate super-resolved images (this will handle wavelet conversion internally)
        sr_images, _ = self.sample_conditional(
            conditioning=lr_images,
            x_image_size=hr_images.shape[-2],
            y_image_size=hr_images.shape[-1],
            steps=1,
            use_ema=self.use_ema
        )

        sr_images = self.adaptive_instance_normalization(sr_images, lr_images)
        
        # All images should now be in image domain
        # Normalize all images
        lr_norm = self.normalize_for_vis(lr_images)
        hr_norm = self.normalize_for_vis(hr_images)
        sr_norm = self.normalize_for_vis(sr_images)
        
        # Create comparison grids
        lr_grid = make_grid(lr_norm, nrow=2, normalize=False, padding=2)
        hr_grid = make_grid(hr_norm, nrow=2, normalize=False, padding=2)
        sr_grid = make_grid(sr_norm, nrow=2, normalize=False, padding=2)
        
        # Log to tensorboard
        if hasattr(self.logger, 'experiment'):
            step = self.trainer.global_step
            self.logger.experiment.add_image(f'{stage}/low_res_input', lr_grid, step)
            self.logger.experiment.add_image(f'{stage}/high_res_target', hr_grid, step)
            self.logger.experiment.add_image(f'{stage}/super_resolved', sr_grid, step)
    

    def normalize_for_vis(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images to [0, 1] range for visualization"""
        images = images.clone()
        images = (images + 1.0) / 2.0
        return images.clamp(0.0, 1.0)

    def on_validation_epoch_end(self):
        """Log samples at end of validation epoch"""
        if hasattr(self.logger, 'experiment'):
            generator = torch.Generator(device=self.device).manual_seed(42)
            self.log_conditional_samples(generator)

    @torch.no_grad()
    def log_conditional_samples(self, generator):
        """Log conditional samples"""
        if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders:
            val_dataloader = self.trainer.val_dataloaders[0] if isinstance(self.trainer.val_dataloaders, list) else self.trainer.val_dataloaders
        elif hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            val_dataloader = self.trainer.datamodule.val_dataloader()
        
        val_batch = next(iter(val_dataloader))
        lr_images = val_batch['lr'][:8]
        hr_images = val_batch['hr'][:8]
        
        # Generate super-resolved images
        sr_images, _ = self.sample_conditional(
            conditioning=lr_images,
            x_image_size=hr_images.shape[-2],
            y_image_size=hr_images.shape[-1],
            steps=10,#instead of 1
            generator=generator,
            use_ema=self.use_ema
        )

        sr_images = sr_images.detach().cpu()
        sr_images = self.adaptive_instance_normalization(sr_images, lr_images)
        
        # Create comparison
        lr_norm = self.normalize_for_vis(lr_images)
        hr_norm = self.normalize_for_vis(hr_images)
        sr_norm = self.normalize_for_vis(sr_images.detach().cpu())
        
        comparison = torch.cat([lr_norm, sr_norm, hr_norm], dim=-1)
        comparison_grid = make_grid(comparison, nrow=2, normalize=False, padding=2)
        
        self.logger.experiment.add_image(
            'samples/conditional_lr_sr_hr_comparison',
            comparison_grid,
            self.trainer.current_epoch
        )

    ## This may not be needed for the SR training, but leaving it if necessary later. ##
    
    # @torch.no_grad()
    # def sample(
    #     self,
    #     num_samples: Optional[int] = 16,
    #     steps: Optional[int] = 1,
    #     x_image_size: Optional[int] = None,
    #     y_image_size: Optional[int] = None,
    #     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    #     use_ema: Optional[bool] = False,
    # ) -> torch.Tensor:
    #     """ Unconditional sampler for testing purposes """

    #     if x_image_size and y_image_size is not None:
    #         shape = (num_samples, self.config.in_channels, x_image_size, y_image_size)
    #     else:
    #         shape = (num_samples, self.config.in_channels, self.config.sample_dimension[0], self.config.sample_dimension[1])

    #     time = torch.tensor([self.time_max], device=self.device)

    #     images: torch.Tensor = self._forward_conditional(
    #         self.model_ema if use_ema else self.model,
    #         randn_tensor(shape, generator=generator, device=self.device) * time,
    #         time,
    #         conditioning=None  # No conditioning for unconditional sampling
    #     )

    #     if steps <= 1:
    #         return images

    #     _timesteps = list(
    #         reversed(range(0, self.bins_max, self.bins_max // steps - 1))
    #     )[1:]
    #     _timesteps = [t + self.bins_max // ((steps - 1) * 2) for t in _timesteps]

    #     times = self.timesteps_to_times(
    #         torch.tensor(_timesteps, device=self.device), bins=150
    #     )

    #     for time in times:
    #         noise = randn_tensor(shape, generator=generator, device=self.device)
    #         images = images + math.sqrt(time.item() ** 2 - self.time_min**2) * noise
    #         images = self._forward_conditional(
    #             self.model_ema if use_ema else self.model,
    #             images,
    #             time[None],
    #             conditioning=None
    #         )

    #     return images


    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Log bins and learning rate
        self._bins_tracker(self.bins)
        self.log("bins", self._bins_tracker, on_epoch=True, logger=True)
        
        if hasattr(self.logger, 'experiment'):
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.logger.experiment.add_scalar(
                'learning_rate', 
                current_lr, 
                self.trainer.current_epoch
            )

    def load_noise_schedule_from_checkpoint(self, checkpoint_path):
        """Load noise schedule parameters from diffusion checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'sqrt_alphas_cumprod_prev' in checkpoint:
                self.register_buffer('sqrt_alphas_cumprod_prev', checkpoint['sqrt_alphas_cumprod_prev'])
            elif 'state_dict' in checkpoint:
                for key, value in checkpoint['state_dict'].items():
                    if 'sqrt_alphas_cumprod_prev' in key:
                        self.register_buffer('sqrt_alphas_cumprod_prev', value)
        except Exception as e:
            print(f"Could not load noise schedule: {e}")


    def times_to_noise_levels(self, times):
        """Convert consistency model times to diffusion model noise levels"""
        # Map consistency times [time_min, time_max] to timestep indices [0, num_timesteps]
        normalized_times = (times.squeeze(-1) - self.time_min) / (self.time_max - self.time_min)
        timestep_indices = (normalized_times * (len(self.sqrt_alphas_cumprod_prev) - 1)).long().clamp(0, len(self.sqrt_alphas_cumprod_prev) - 1)
        
        # Get corresponding noise levels
        noise_levels = self.sqrt_alphas_cumprod_prev[timestep_indices]
        return noise_levels.unsqueeze(-1)  # [batch_size, 1] format

    def calc_mean_std(self, feat: Tensor, eps=1e-5):
        """Calculate mean and std for adaptive_instance_normalization.
        Args:
            feat (Tensor): 4D tensor.
            eps (float): A small value added to the variance to avoid
                divide-by-zero. Default: 1e-5.
        """
        # print("Feat shape: ", feat.shape)
        # feat=feat.transpose((2,0,1))
        size = feat.size()
        # print("Tensor size: ", size)
        if len(size) == 3:
            feat = feat.unsqueeze(0)
        assert len(feat.shape) == 4, 'The input feature should be 4D tensor.'
        b, c = size[:2]
        feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
        feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat:Tensor, style_feat:Tensor):
        """Adaptive instance normalization.
        Adjust the reference features to have the similar color and illuminations
        as those in the degradate features.
        Args:
            content_feat (Tensor): The reference feature.
            style_feat (Tensor): The degradate features.
        """
        if len(content_feat.size()) == 3:
            content_feat=content_feat.unsqueeze(0)
            style_feat=style_feat.unsqueeze(0)
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


    def configure_metrics(self, use_lpips=False, use_clip=False):
        """
        Configure which metrics to use during validation
        
        Args:
            use_lpips: Whether to calculate LPIPS (slower)
            use_clip: Whether to calculate CLIP score (slower)
        """
        self.lpips_model = 'alexnet' if use_lpips else None
        self.clip_model = 'clip-ViT-B/16' if use_clip else None
        print(f"Metrics configured - LPIPS: {use_lpips}, CLIP: {use_clip}")


    def on_save_checkpoint(self, checkpoint):
        """Save additional state for proper resumption"""
        # Save EMA model states (CRITICAL - not saved by default)
        checkpoint['model_ema_state_dict'] = self.model_ema.state_dict()
        checkpoint['dwsr_ema_state_dict'] = self.dwsr_ema.state_dict()
        
        # Save training progress info
        checkpoint['bins'] = self.bins
        checkpoint['global_step'] = self.trainer.global_step
        checkpoint['current_epoch'] = self.trainer.current_epoch
        
        # Save metric tracker states (optional but recommended)
        checkpoint['train_loss_tracker'] = self._train_loss_tracker.state_dict()
        checkpoint['val_loss_tracker'] = self._val_loss_tracker.state_dict()
        checkpoint['val_psnr_tracker'] = self._val_psnr_tracker.state_dict()
        checkpoint['val_ssim_tracker'] = self._val_ssim_tracker.state_dict()
        
        # Save noise schedule if it exists
        if hasattr(self, 'sqrt_alphas_cumprod_prev'):
            checkpoint['sqrt_alphas_cumprod_prev'] = self.sqrt_alphas_cumprod_prev
        
        print(f"Saving checkpoint at step {self.trainer.global_step}, bins: {self.bins}")
        return checkpoint


    def on_load_checkpoint(self, checkpoint):
        """Load additional state for proper resumption"""
        # Load EMA model states
        if 'model_ema_state_dict' in checkpoint:
            self.model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
            print("✓ Loaded model_ema from checkpoint")
        else:
            print("⚠ Warning: model_ema state not found in checkpoint")
        
        if 'dwsr_ema_state_dict' in checkpoint:
            self.dwsr_ema.load_state_dict(checkpoint['dwsr_ema_state_dict'])
            print("✓ Loaded dwsr_ema from checkpoint")
        else:
            print("⚠ Warning: dwsr_ema state not found in checkpoint")
        
        # Load metric trackers (optional)
        if 'train_loss_tracker' in checkpoint:
            self._train_loss_tracker.load_state_dict(checkpoint['train_loss_tracker'])
        if 'val_loss_tracker' in checkpoint:
            self._val_loss_tracker.load_state_dict(checkpoint['val_loss_tracker'])
        if 'val_psnr_tracker' in checkpoint:
            self._val_psnr_tracker.load_state_dict(checkpoint['val_psnr_tracker'])
        if 'val_ssim_tracker' in checkpoint:
            self._val_ssim_tracker.load_state_dict(checkpoint['val_ssim_tracker'])
        
        # Load noise schedule
        if 'sqrt_alphas_cumprod_prev' in checkpoint:
            self.register_buffer('sqrt_alphas_cumprod_prev', checkpoint['sqrt_alphas_cumprod_prev'])
        
        # Print resumption info
        if 'bins' in checkpoint:
            print(f"Resuming from bins: {checkpoint['bins']}")
        if 'global_step' in checkpoint:
            print(f"Resuming from global step: {checkpoint['global_step']}")
        if 'current_epoch' in checkpoint:
            print(f"Resuming from epoch: {checkpoint['current_epoch']}")