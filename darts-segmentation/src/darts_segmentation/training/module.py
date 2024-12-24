"""Training script for DARTS segmentation."""

from pathlib import Path

import lightning as L  # noqa: N812
import matplotlib.pyplot as plt
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
    """Lightning module for training a segmentation model using the segmentation_models_pytorch library."""

    def __init__(
        self,
        config: SMPSegmenterConfig,
        learning_rate: float = 1e-5,
        gamma: float = 0.9,
        focal_loss_alpha: float | None = None,
        focal_loss_gamma: float = 2.0,
        plot_every_n_val_epochs: int = 5,
    ):
        """Initialize the SMPSegmenter.

        Args:
            config (SMPSegmenterConfig): Configuration for the segmentation model.
            learning_rate (float, optional): Initial learning rate. Defaults to 1e-5.
            gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.9.
            focal_loss_alpha (float, optional): Weight factor to balance positive and negative samples.
                Alpha must be in [0...1] range, high values will give more weight to positive class.
                None will not weight samples. Defaults to None.
            focal_loss_gamma (float, optional): Focal loss power factor. Defaults to 2.0.
            plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.

        """
        super().__init__()
        # This saves config, learning_rate and gamma under self.hparams
        self.save_hyperparameters(ignore=["plot_every_n_val_epochs"])
        self.model = smp.create_model(**config["model"], activation="sigmoid")

        # Assumes that the training preparation was done with setting invalid pixels in the mask to 2
        self.loss_fn = smp.losses.FocalLoss(
            mode="binary", alpha=focal_loss_alpha, gamma=focal_loss_gamma, ignore_index=2
        )

        metric_kwargs = {"task": "binary", "validate_args": False, "ignore_index": 2}
        metrics = MetricCollection(
            {
                "Accuracy": Accuracy(**metric_kwargs),
                "Precision": Precision(**metric_kwargs),
                "Specificity": Specificity(**metric_kwargs),
                "Recall": Recall(**metric_kwargs),
                "F1Score": F1Score(**metric_kwargs),
                "JaccardIndex": JaccardIndex(**metric_kwargs),
                "CohenKappa": CohenKappa(**metric_kwargs),
                "HammingDistance": HammingDistance(**metric_kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.val_metrics.add_metrics(
            {
                "AUROC": AUROC(thresholds=20, **metric_kwargs),
                "AveragePrecision": AveragePrecision(thresholds=20, **metric_kwargs),
            }
        )
        self.val_roc = ROC(thresholds=20, **metric_kwargs)
        self.val_prc = PrecisionRecallCurve(thresholds=20, **metric_kwargs)
        self.val_cmx = ConfusionMatrix(normalize="true", **metric_kwargs)
        self.plot_every_n_val_epochs = plot_every_n_val_epochs

    def __repr__(self):  # noqa: D105
        return f"SMPSegmenter({self.hparams['config']['model']})"

    @property
    def is_val_plot_epoch(self):
        """Check if the current epoch is an epoch where validation samples should be plotted.

        Returns:
            bool: True if the current epoch is a plot epoch, False otherwise.

        """
        n = self.plot_every_n_val_epochs * self.trainer.check_val_every_n_epoch
        return ((self.current_epoch + 1) % n) == 0 or self.current_epoch == 0

    def training_step(self, batch, batch_idx):  # noqa: D102
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        loss = self.loss_fn(y_hat, y.long())
        self.train_metrics(y_hat, y)
        self.log("train/loss", loss)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):  # noqa: D102
        self.train_metrics.reset()
        self.log("learning_rate", self.lr_schedulers().get_last_lr()[0])

    def validation_step(self, batch, batch_idx):  # noqa: D102
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        loss = self.loss_fn(y_hat, y.long())
        self.log("val/loss", loss)

        self.val_metrics.update(y_hat, y)
        self.val_roc.update(y_hat, y)
        self.val_prc.update(y_hat, y)
        self.val_cmx.update(y_hat, y)

        # Create figures for the samples (plot at maximum 24)
        is_last_batch = self.trainer.num_val_batches == (batch_idx + 1)
        max_batch_idx = 24 // x.shape[0]  # Does only work if NOT last batch, since last batch may be smaller
        # If num_val_batches is 1 then this batch is the last one, but we still want to log it. despite its size
        # Does not work well for batch-sizes larger than 24!
        should_log_batch = (max_batch_idx >= batch_idx and not is_last_batch) or self.trainer.num_val_batches == 1
        if self.is_val_plot_epoch and should_log_batch:
            for i in range(x.shape[0]):
                fig, _ = plot_sample(x[i], y[i], y_hat[i], self.hparams.config["input_combination"])
                for logger in self.loggers:
                    if isinstance(logger, CSVLogger):
                        fig_dir = Path(logger.log_dir) / "figures"
                        fig_dir.mkdir(exist_ok=True)
                        fig.savefig(fig_dir / f"sample_{self.global_step}_{batch_idx}_{i}.png")
                    if isinstance(logger, WandbLogger):
                        wandb_run: Run = logger.experiment
                        # We don't commit the log yet, so that the step is increased with the next lightning log
                        # Which happens at the end of the validation epoch
                        wandb_run.log({f"val-samples/sample_{batch_idx}_{i}": wandb.Image(fig)}, commit=False)
                fig.clear()
                plt.close(fig)

        return loss

    def on_validation_epoch_end(self):  # noqa: D102
        # Only do this every self.plot_every_n_val_epochs epochs
        if self.is_val_plot_epoch:
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
                    wandb_run.log({"val/cmx": wandb.Image(fig_cmx)}, commit=False)
                    wandb_run.log({"val/roc": wandb.Image(fig_roc)}, commit=False)
                    wandb_run.log({"val/prc": wandb.Image(fig_prc)}, commit=False)

            fig_cmx.clear()
            fig_roc.clear()
            fig_prc.clear()
            plt.close("all")

        # This will also commit the accumulated plots
        self.log_dict(self.val_metrics.compute())

        self.val_metrics.reset()
        self.val_roc.reset()
        self.val_prc.reset()
        self.val_cmx.reset()

    def configure_optimizers(self):  # noqa: D102
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
