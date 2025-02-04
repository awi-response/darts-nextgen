"""PyTorch Lightning Callbacks for training and validation."""

from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
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

from darts_segmentation.training.viz import plot_sample


class ValidationCallback(Callback):
    """Callback for validation metrics and visualizations."""

    trainer: Trainer
    pl_module: LightningModule

    def __init__(self, *, input_combination: list[str], val_set: str = "val", plot_every_n_val_epochs: int = 5):
        """Initialize the ValidationCallback.

        Args:
            input_combination (list[str]): List of input names to combine for the visualization.
            val_set (str, optional): Name of the validation set. Only used for naming the metrics. Defaults to "val".
            plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.

        """
        assert "/" not in val_set, "val_set must not contain '/'"
        self.val_set = val_set
        self.plot_every_n_val_epochs = plot_every_n_val_epochs
        self.input_combination = input_combination

    def is_val_plot_epoch(self, current_epoch: int, check_val_every_n_epoch: int | None):
        """Check if the current epoch is an epoch where validation samples should be plotted.

        Args:
            current_epoch (int): The current epoch.
            check_val_every_n_epoch (int | None): The number of epochs to check for plotting.
                If None, no plotting is done.

        Returns:
            bool: True if the current epoch is a plot epoch, False otherwise.

        """
        if check_val_every_n_epoch is None:
            return False
        n = self.plot_every_n_val_epochs * check_val_every_n_epoch
        return ((current_epoch + 1) % n) == 0 or current_epoch == 0

    def setup(self, trainer, pl_module):  # noqa: D102
        # Save references to the trainer and pl_module
        self.trainer = trainer
        self.pl_module = pl_module

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
        self.val_metrics = metrics.clone(prefix=f"{self.val_set}/")
        self.val_metrics.add_metrics(
            {
                "AUROC": AUROC(thresholds=20, **metric_kwargs),
                "AveragePrecision": AveragePrecision(thresholds=20, **metric_kwargs),
            }
        )
        self.val_roc = ROC(thresholds=20, **metric_kwargs)
        self.val_prc = PrecisionRecallCurve(thresholds=20, **metric_kwargs)
        self.val_cmx = ConfusionMatrix(normalize="true", **metric_kwargs)

    def teardown(self, trainer, pl_module, stage):  # noqa: D102
        # Delete the references to the trainer and pl_module
        del self.trainer
        del self.pl_module

        # Delete the metrics
        del self.val_metrics
        del self.val_roc
        del self.val_prc
        del self.val_cmx

        return super().teardown(trainer, pl_module, stage)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: D102
        x, y = batch
        # Expect the output to has a tensor called "y_hat"
        assert "y_hat" in outputs, (
            "Output does not contain 'y_hat' tensor."
            " Please make sure the 'validation_step' method returns a dict with 'y_hat' and 'loss' keys."
            " The 'y_hat' should be the model's prediction (a pytorch tensor of shape [B, C, H, W])."
            " The 'loss' should be the loss value (a scalar tensor).",
        )
        y_hat = outputs["y_hat"]

        self.val_metrics.update(y_hat, y)
        self.val_roc.update(y_hat, y)
        self.val_prc.update(y_hat, y)
        self.val_cmx.update(y_hat, y)

        # Create figures for the samples (plot at maximum 24)
        is_last_batch = trainer.num_val_batches == (batch_idx + 1)
        max_batch_idx = (24 // x.shape[0]) - 1  # Does only work if NOT last batch, since last batch may be smaller
        # If num_val_batches is 1 then this batch is the last one, but we still want to log it. despite its size
        # Will plot the first 24 samples of the first batch if batch-size is larger than 24
        should_log_batch = (
            (max_batch_idx >= batch_idx and not is_last_batch)
            or trainer.num_val_batches == 1
            or (max_batch_idx == -1 and batch_idx == 0)
        )
        is_val_plot_epoch = self.is_val_plot_epoch(pl_module.current_epoch, trainer.check_val_every_n_epoch)
        if is_val_plot_epoch and should_log_batch:
            for i in range(min(x.shape[0], 24)):
                fig, _ = plot_sample(x[i], y[i], y_hat[i], self.input_combination)
                for logger in pl_module.loggers:
                    if isinstance(logger, CSVLogger):
                        fig_dir = Path(logger.log_dir) / "figures" / f"{self.val_set}-samples"
                        fig_dir.mkdir(exist_ok=True, parents=True)
                        fig.savefig(fig_dir / f"sample_{pl_module.global_step}_{batch_idx}_{i}.png")
                    if isinstance(logger, WandbLogger):
                        wandb_run: Run = logger.experiment
                        # We don't commit the log yet, so that the step is increased with the next lightning log
                        # Which happens at the end of the validation epoch
                        img_name = f"{self.val_set}-samples/sample_{batch_idx}_{i}"
                        wandb_run.log({img_name: wandb.Image(fig)}, commit=False)
                fig.clear()
                plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module):  # noqa: D102
        # Only do this every self.plot_every_n_val_epochs epochs
        is_val_plot_epoch = self.is_val_plot_epoch(pl_module.current_epoch, trainer.check_val_every_n_epoch)
        if is_val_plot_epoch:
            self.val_cmx.compute()
            self.val_roc.compute()
            self.val_prc.compute()

            # Plot roc, prc and confusion matrix to disk and wandb
            fig_cmx, _ = self.val_cmx.plot(cmap="Blues")
            fig_roc, _ = self.val_roc.plot(score=True)
            fig_prc, _ = self.val_prc.plot(score=True)

            # Check for a wandb or csv logger to log the images
            for logger in pl_module.loggers:
                if isinstance(logger, CSVLogger):
                    fig_dir = Path(logger.log_dir) / "figures" / f"{self.val_set}-samples"
                    fig_dir.mkdir(exist_ok=True, parents=True)
                    fig_cmx.savefig(fig_dir / f"cmx_{pl_module.global_step}png")
                    fig_roc.savefig(fig_dir / f"roc_{pl_module.global_step}png")
                    fig_prc.savefig(fig_dir / f"prc_{pl_module.global_step}.png")
                if isinstance(logger, WandbLogger):
                    wandb_run: Run = logger.experiment
                    wandb_run.log({f"{self.val_set}/cmx": wandb.Image(fig_cmx)}, commit=False)
                    wandb_run.log({f"{self.val_set}/roc": wandb.Image(fig_roc)}, commit=False)
                    wandb_run.log({f"{self.val_set}/prc": wandb.Image(fig_prc)}, commit=False)

            fig_cmx.clear()
            fig_roc.clear()
            fig_prc.clear()
            plt.close("all")

        # This will also commit the accumulated plots
        pl_module.log_dict(self.val_metrics.compute())

        self.val_metrics.reset()
        self.val_roc.reset()
        self.val_prc.reset()
        self.val_cmx.reset()
