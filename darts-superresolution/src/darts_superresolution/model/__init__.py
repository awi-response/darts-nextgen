import logging

from darts_superresolution.model.inference import ModelConfig as ModelConfig
from darts_superresolution.model.inference import define_net as define_net
from darts_superresolution.model.wave_modules.diffusion import GaussianDiffusion as GaussianDiffusion

logger = logging.getLogger(__name__)


def create_model(opt, device):
    from model.model import DDPM as M

    m = M(opt, device)
    logger.info(f"Model [{m.__class__.__name__:s}] is created.")
    return m
