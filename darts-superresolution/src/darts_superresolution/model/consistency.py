import copy
import math
import os
from typing import List, Optional, Type, Union
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim, Tensor
from torchvision.utils import make_grid
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
from model import unet
from model.diffusion import DWSR
from config.model_parameters import ConsistencyConfig


class ConsistencyWavelet(LightningModule):
    """ Consistency model with wavelet transform integration """

    def __init__(
        self,
        config: Type[ConsistencyConfig],
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

        # # Add noise schedule parameters (matching diffusion model)
        # self.register_buffer('sqrt_alphas_cumprod_prev', torch.tensor([1.0]))  # Placeholder
        # if diffusion_checkpoint_path and os.path.exists(diffusion_checkpoint_path):
        #     self.load_noise_schedule_from_checkpoint(diffusion_checkpoint_path)

        # Add noise schedule parameters (matching diffusion model)
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


    def normalize_for_vis(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images to [0, 1] range for visualization"""
        images = images.clone()
        images = (images + 1.0) / 2.0
        return images.clamp(0.0, 1.0)

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
        size = feat.size()

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