# ruff: noqa: D100
# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D105
# ruff: noqa: D107

"""Training script for DARTS segmentation."""

import lightning as L  # noqa: N812
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from torchmetrics import (
    Accuracy,
    CohenKappa,
    F1Score,
    HammingDistance,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
    Specificity,
)

from darts_segmentation.segment import SMPSegmenterConfig


class SMPSegmenter(L.LightningModule):
    def __init__(self, config: SMPSegmenterConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(**config["model"])

        metrics = MetricCollection(
            {
                "Accuracy": Accuracy(task="binary", validate_args=False),
                "Precision": Precision(task="binary", validate_args=False),
                "Specificity": Specificity(task="binary", validate_args=False),
                "Recall": Recall(task="binary", validate_args=False),
                "F1Score": F1Score(task="binary", validate_args=False),
                "JaccardIndex": JaccardIndex(task="binary", validate_args=False),
                "CohenKappa": CohenKappa(task="binary", validate_args=False),
                "HammingDistance": HammingDistance(task="binary", validate_args=False),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y.unsqueeze(1).float())
        self.train_metrics(y_hat, y)
        self.log("train_loss", loss)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y.unsqueeze(1).float())
        self.log("val_loss", loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
