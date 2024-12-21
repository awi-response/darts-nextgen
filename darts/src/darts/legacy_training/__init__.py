"""Legacy training module for DARTS."""

from darts.legacy_training.preprocess import preprocess_s2_train_data as preprocess_s2_train_data
from darts.legacy_training.train import sweep_smp as sweep_smp
from darts.legacy_training.train import train_smp as train_smp
from darts.legacy_training.util import convert_lightning_checkpoint as convert_lightning_checkpoint
