"""Legacy training module for DARTS."""

from darts.legacy_training.preprocess.planet import preprocess_planet_train_data as preprocess_planet_train_data
from darts.legacy_training.preprocess.s2 import preprocess_s2_train_data as preprocess_s2_train_data

# from darts.legacy_training.train import optuna_sweep_smp as optuna_sweep_smp
from darts.legacy_training.sweep import optuna_sweep_smp as optuna_sweep_smp
from darts.legacy_training.test import test_smp as test_smp
from darts.legacy_training.train import train_smp as train_smp
from darts.legacy_training.train import wandb_sweep_smp as wandb_sweep_smp
from darts.legacy_training.util import convert_lightning_checkpoint as convert_lightning_checkpoint
