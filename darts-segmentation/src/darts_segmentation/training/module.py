# ruff: noqa: D100
# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D105
# ruff: noqa: D107

"""Training script for DARTS segmentation."""

from pathlib import Path

import lightning as L  # noqa: N812
import segmentation_models_pytorch as smp
import torch.optim as optim
import wandb
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchmetrics import (
    AUROC,
    ROC,
    Accuracy,
    AveragePrecision,
    CohenKappa,
    ConfusionMatrix,
    F1Score,
    HammingDistance,
    JaccardIndex,
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    Recall,
    Specificity,
)
from wandb.sdk.wandb_run import Run

from darts_segmentation.segment import SMPSegmenterConfig
from darts_segmentation.training.viz import plot_sample


class SMPSegmenter(L.LightningModule):
    def __init__(self, config: SMPSegmenterConfig, learning_rate: float = 1e-5, gamma: float = 0.9):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(**config["model"], activation="sigmoid")

        self.loss_fn = smp.losses.FocalLoss(mode="binary")

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
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.val_metrics.add_metrics(
            {
                "AUROC": AUROC(task="binary", thresholds=20, validate_args=False),
                "AveragePrecision": AveragePrecision(task="binary", thresholds=20, validate_args=False),
            }
        )
        self.val_roc = ROC(task="binary", thresholds=20, validate_args=False)
        self.val_prc = PrecisionRecallCurve(task="binary", thresholds=20, validate_args=False)
        self.val_cmx = ConfusionMatrix(task="binary", normalize="true", validate_args=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        loss = self.loss_fn(y_hat, y.long())
        self.train_metrics(y_hat, y)
        self.log("train/loss", loss)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        loss = self.loss_fn(y_hat, y.long())
        self.log("val/loss", loss)

        self.val_metrics.update(y_hat, y)
        self.val_roc.update(y_hat, y)
        self.val_prc.update(y_hat, y)
        self.val_cmx.update(y_hat, y)

        # Create figures for the samples
        for i in range(x.shape[0]):
            fig, _ = plot_sample(x[i], y[i], y_hat[i], self.hparams.config["input_combination"])
            for logger in self.loggers:
                if isinstance(logger, CSVLogger):
                    fig_dir = Path(logger.log_dir) / "figures"
                    fig_dir.mkdir(exist_ok=True)
                    fig.savefig(fig_dir / f"sample_{self.global_step}_{batch_idx}_{i}.png")
                if isinstance(logger, WandbLogger):
                    wandb_run: Run = logger.experiment
                    wandb_run.log({f"val/sample_{batch_idx}_{i}": wandb.Image(fig)}, step=self.global_step)
            fig.clear()

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())

        self.val_cmx.compute()
        self.val_roc.compute()
        self.val_prc.compute()

        # Plot roc, prc and confusion matrix to disk and wandb
        fig_cmx, _ = self.val_cmx.plot(cmap="Blues")
        fig_roc, _ = self.val_roc.plot(score=True)
        fig_prc, _ = self.val_prc.plot(score=True)

        # Check for a wandb or csv logger to log the images
        for logger in self.loggers:
            if isinstance(logger, CSVLogger):
                fig_dir = Path(logger.log_dir) / "figures"
                fig_dir.mkdir(exist_ok=True)
                fig_cmx.savefig(fig_dir / f"cmx_{self.global_step}png")
                fig_roc.savefig(fig_dir / f"roc_{self.global_step}png")
                fig_prc.savefig(fig_dir / f"prc_{self.global_step}.png")
            if isinstance(logger, WandbLogger):
                wandb_run: Run = logger.experiment
                wandb_run.log({"val/cmx": wandb.Image(fig_cmx)}, step=self.global_step)
                wandb_run.log({"val/roc": wandb.Image(fig_roc)}, step=self.global_step)
                wandb_run.log({"val/prc": wandb.Image(fig_prc)}, step=self.global_step)

        fig_cmx.clear()
        fig_roc.clear()
        fig_prc.clear()

        self.val_metrics.reset()
        self.val_roc.reset()
        self.val_prc.reset()
        self.val_cmx.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
