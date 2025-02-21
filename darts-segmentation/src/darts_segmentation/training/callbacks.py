"""PyTorch Lightning Callbacks for training and validation."""

import logging
from pathlib import Path
from typing import Literal

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

from darts_segmentation.metrics import (
    BinaryBoundaryIoU,
    BinaryInstanceAccuracy,
    BinaryInstanceAveragePrecision,
    BinaryInstanceConfusionMatrix,
    BinaryInstanceF1Score,
    BinaryInstancePrecision,
    BinaryInstancePrecisionRecallCurve,
    BinaryInstanceRecall,
)
from darts_segmentation.training.viz import plot_sample

logger = logging.getLogger(__name__.replace("darts_", "darts."))

Stage = Literal["fit", "validate", "test", "predict"]


class BinarySegmentationMetrics(Callback):
    """Callback for validation metrics and visualizations."""

    trainer: Trainer
    pl_module: LightningModule
    stage: Stage

    train_metrics: MetricCollection
    val_metrics: MetricCollection
    val_roc: ROC
    val_prc: PrecisionRecallCurve
    val_cmx: ConfusionMatrix
    test_metrics: MetricCollection
    test_roc: ROC
    test_prc: PrecisionRecallCurve
    test_cmx: ConfusionMatrix
    test_instance_prc: BinaryInstancePrecisionRecallCurve
    test_instance_cmx: BinaryInstanceConfusionMatrix

    def __init__(
        self,
        *,
        input_combination: list[str],
        val_set: str = "val",
        test_set: str = "test",
        plot_every_n_val_epochs: int = 5,
        is_crossval: bool = False,
    ):
        """Initialize the ValidationCallback.

        Args:
            input_combination (list[str]): List of input names to combine for the visualization.
            val_set (str, optional): Name of the validation set. Only used for naming the validation metrics.
                Defaults to "val".
            test_set (str, optional): Name of the test set. Only used for naming the test metrics. Defaults to "test".
            plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.
            is_crossval (bool, optional): Whether the training is done with cross-validation.
                This will change the logging behavior of scalar metrics from logging to {val_set} to just "val".
                The logging behaviour of the samples is not affected.
                Defaults to False.

        """
        assert "/" not in val_set, "val_set must not contain '/'"
        assert "/" not in test_set, "test_set must not contain '/'"
        self.val_set = val_set
        self.test_set = test_set
        self.plot_every_n_val_epochs = plot_every_n_val_epochs
        self.input_combination = input_combination
        self.is_crossval = is_crossval

    @property
    def _val_prefix(self):
        return "val" if self.is_crossval else self.val_set

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

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Stage):
        """Setups the callback.

        Creates metrics required for the specific stage:

        - For the "fit" stage, creates training and validation metrics and visualizations.
        - For the "validate" stage, only creates validation metrics and visualizations.
        - For the "test" stage, only creates test metrics and visualizations.
        - For the "predict" stage, no metrics or visualizations are created.

        Always maps the trainer and pl_module to the callback.

        Training and validation metrics are "simple" metrics from torchmetrics.
        The validation visualizations are more complex metrics from torchmetrics.
        The test metrics and vsiualizations are the same as the validation ones,
        and also include custom "Instance" metrics.

        Args:
            trainer (Trainer): The lightning trainer.
            pl_module (LightningModule): The lightning module.
            stage (Literal["fit", "validate", "test", "predict"]): The current stage.
                One of: "fit", "validate", "test", "predict".

        """
        # Save references to the trainer and pl_module
        self.trainer = trainer
        self.pl_module = pl_module
        self.stage = stage

        # We don't want to use memory in the predict stage
        if stage == "predict":
            return

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

        added_metrics: list[str] = []

        # Train metrics only for the fit stage
        if stage == "fit":
            pl_module.train_metrics = metrics.clone(prefix="train/")
            added_metrics += list(pl_module.train_metrics.keys())
        # Validation metrics and visualizations for the fit and validate stages
        if stage == "fit" or stage == "validate":
            pl_module.val_metrics = metrics.clone(prefix=f"{self._val_prefix}/")
            pl_module.val_metrics.add_metrics(
                {
                    "AUROC": AUROC(thresholds=20, **metric_kwargs),
                    "AveragePrecision": AveragePrecision(thresholds=20, **metric_kwargs),
                }
            )
            pl_module.val_roc = ROC(thresholds=20, **metric_kwargs)
            pl_module.val_prc = PrecisionRecallCurve(thresholds=20, **metric_kwargs)
            pl_module.val_cmx = ConfusionMatrix(normalize="true", **metric_kwargs)
            added_metrics += list(pl_module.val_metrics.keys())
            added_metrics += [f"{self._val_prefix}/{m}" for m in ["roc", "prc", "cmx"]]

        # Test metrics and visualizations for the test stage
        if stage == "test":
            pl_module.test_metrics = metrics.clone(prefix=f"{pl_module.test_set}/")
            pl_module.test_metrics.add_metrics(
                {
                    "AUROC": AUROC(thresholds=20, **metric_kwargs),
                    "AveragePrecision": AveragePrecision(thresholds=20, **metric_kwargs),
                }
            )
            pl_module.test_roc = ROC(thresholds=20, **metric_kwargs)
            pl_module.test_prc = PrecisionRecallCurve(thresholds=20, **metric_kwargs)
            pl_module.test_cmx = ConfusionMatrix(normalize="true", **metric_kwargs)

            # Instance Metrics
            instance_metric_kwargs = {"validate_args": False, "ignore_index": 2, "matching_threshold": 0.3}
            pl_module.test_metrics.add_metrics(
                {
                    "InstanceAccuracy": BinaryInstanceAccuracy(**instance_metric_kwargs),
                    "InstancePrecision": BinaryInstancePrecision(**instance_metric_kwargs),
                    "InstanceRecall": BinaryInstanceRecall(**instance_metric_kwargs),
                    "InstanceF1Score": BinaryInstanceF1Score(**instance_metric_kwargs),
                    "InstanceAveragePrecision": BinaryInstanceAveragePrecision(thresholds=20, **instance_metric_kwargs),
                }
            )
            boundary_metric_kwargs = {"validate_args": False, "ignore_index": 2}
            pl_module.test_metrics.add_metrics(
                {
                    "InstanceBoundaryIoU": BinaryBoundaryIoU(**boundary_metric_kwargs),
                }
            )
            pl_module.test_instance_prc = BinaryInstancePrecisionRecallCurve(thresholds=20, **instance_metric_kwargs)
            pl_module.test_instance_cmx = BinaryInstanceConfusionMatrix(normalize=True, **instance_metric_kwargs)

            added_metrics += list(pl_module.test_metrics.keys())
            added_metrics += [f"{self.test_set}/{m}" for m in ["roc", "prc", "cmx", "instance_prc", "instance_cmx"]]

        # Log the added metrics
        sep = "\n\t- "
        logger.debug(f"Added metrics:{sep + sep.join(added_metrics)}")

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: Stage):  # noqa: D102
        # Delete the references to the trainer and pl_module
        del self.trainer
        del self.pl_module
        del self.stage

        # No need to delete anything if we are in the predict stage
        if stage == "predict":
            return

        if stage == "fit":
            del pl_module.train_metrics

        if stage == "fit" or stage == "validate":
            del pl_module.val_metrics
            del pl_module.val_roc
            del pl_module.val_prc
            del pl_module.val_cmx

        if stage == "test":
            del pl_module.test_metrics
            del pl_module.test_roc
            del pl_module.test_prc
            del pl_module.test_cmx

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):  # noqa: D102
        pl_module.log("train/loss", outputs["loss"])
        _, y = batch
        # Expect the output to has a tensor called "y_hat"
        assert "y_hat" in outputs, (
            "Output does not contain 'y_hat' tensor."
            " Please make sure the 'training_step' method returns a dict with 'y_hat' and 'loss' keys."
            " The 'y_hat' should be the model's prediction (a pytorch tensor of shape [B, C, H, W])."
            " The 'loss' should be the loss value (a scalar tensor).",
        )
        y_hat = outputs["y_hat"]
        pl_module.train_metrics(y_hat, y)
        pl_module.log_dict(pl_module.train_metrics, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        pl_module.train_metrics.reset()

    def on_validation_batch_end(  # noqa: D102
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pl_module.log(f"{self._val_prefix}/loss", outputs["loss"])
        x, y = batch
        # Expect the output to has a tensor called "y_hat"
        assert "y_hat" in outputs, (
            "Output does not contain 'y_hat' tensor."
            " Please make sure the 'validation_step' method returns a dict with 'y_hat' and 'loss' keys."
            " The 'y_hat' should be the model's prediction (a pytorch tensor of shape [B, C, H, W])."
            " The 'loss' should be the loss value (a scalar tensor).",
        )
        y_hat = outputs["y_hat"]

        pl_module.val_metrics.update(y_hat, y)
        pl_module.val_roc.update(y_hat, y)
        pl_module.val_prc.update(y_hat, y)
        pl_module.val_cmx.update(y_hat, y)

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
                for pllogger in pl_module.loggers:
                    if isinstance(pllogger, CSVLogger):
                        fig_dir = Path(pllogger.log_dir) / "figures" / f"{self.val_set}-samples"
                        fig_dir.mkdir(exist_ok=True, parents=True)
                        fig.savefig(fig_dir / f"sample_{pl_module.global_step}_{batch_idx}_{i}.png")
                    if isinstance(pllogger, WandbLogger):
                        wandb_run: Run = pllogger.experiment
                        # We don't commit the log yet, so that the step is increased with the next lightning log
                        # Which happens at the end of the validation epoch
                        img_name = f"{self.val_set}-samples/sample_{batch_idx}_{i}"
                        wandb_run.log({img_name: wandb.Image(fig)}, commit=False)
                fig.clear()
                plt.close(fig)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        # Only do this every self.plot_every_n_val_epochs epochs
        is_val_plot_epoch = self.is_val_plot_epoch(pl_module.current_epoch, trainer.check_val_every_n_epoch)
        if is_val_plot_epoch:
            pl_module.val_cmx.compute()
            pl_module.val_roc.compute()
            pl_module.val_prc.compute()

            # Plot roc, prc and confusion matrix to disk and wandb
            fig_cmx, _ = pl_module.val_cmx.plot(cmap="Blues")
            fig_roc, _ = pl_module.val_roc.plot(score=True)
            fig_prc, _ = pl_module.val_prc.plot(score=True)

            # Check for a wandb or csv logger to log the images
            for pllogger in pl_module.loggers:
                if isinstance(pllogger, CSVLogger):
                    fig_dir = Path(pllogger.log_dir) / "figures" / f"{self._val_prefix}-samples"
                    fig_dir.mkdir(exist_ok=True, parents=True)
                    fig_cmx.savefig(fig_dir / f"cmx_{pl_module.global_step}png")
                    fig_roc.savefig(fig_dir / f"roc_{pl_module.global_step}png")
                    fig_prc.savefig(fig_dir / f"prc_{pl_module.global_step}.png")
                if isinstance(pllogger, WandbLogger):
                    wandb_run: Run = pllogger.experiment
                    wandb_run.log({f"{self._val_prefix}/cmx": wandb.Image(fig_cmx)}, commit=False)
                    wandb_run.log({f"{self._val_prefix}/roc": wandb.Image(fig_roc)}, commit=False)
                    wandb_run.log({f"{self._val_prefix}/prc": wandb.Image(fig_prc)}, commit=False)

            fig_cmx.clear()
            fig_roc.clear()
            fig_prc.clear()
            plt.close("all")

        # This will also commit the accumulated plots
        pl_module.log_dict(pl_module.val_metrics.compute())

        pl_module.val_metrics.reset()
        pl_module.val_roc.reset()
        pl_module.val_prc.reset()
        pl_module.val_cmx.reset()

    def on_test_batch_end(  # noqa: D102
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pl_module.log(f"{self.test_set}/loss", outputs["loss"])
        x, y = batch
        assert "y_hat" in outputs, (
            "Output does not contain 'y_hat' tensor."
            " Please make sure the 'test_step' method returns a dict with 'y_hat' and 'loss' keys."
            " The 'y_hat' should be the model's prediction (a pytorch tensor of shape [B, C, H, W])."
            " The 'loss' should be the loss value (a scalar tensor).",
        )
        y_hat = outputs["y_hat"]

        pl_module.test_metrics.update(y_hat, y)
        pl_module.test_roc.update(y_hat, y)
        pl_module.test_prc.update(y_hat, y)
        pl_module.test_cmx.update(y_hat, y)
        pl_module.test_instance_prc.update(y_hat, y)
        pl_module.test_instance_cmx.update(y_hat, y)

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
        if should_log_batch:
            for i in range(min(x.shape[0], 24)):
                fig, _ = plot_sample(x[i], y[i], y_hat[i], self.input_combination)
                for pllogger in pl_module.loggers:
                    if isinstance(pllogger, CSVLogger):
                        fig_dir = Path(pllogger.log_dir) / "figures" / f"{self.test_set}-samples"
                        fig_dir.mkdir(exist_ok=True, parents=True)
                        fig.savefig(fig_dir / f"sample_{pl_module.global_step}_{batch_idx}_{i}.png")
                    if isinstance(pllogger, WandbLogger):
                        wandb_run: Run = pllogger.experiment
                        # We don't commit the log yet, so that the step is increased with the next lightning log
                        # Which happens at the end of the validation epoch
                        img_name = f"{self.test_set}-samples/sample_{batch_idx}_{i}"
                        wandb_run.log({img_name: wandb.Image(fig)}, commit=False)
                fig.clear()
                plt.close(fig)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        pl_module.test_cmx.compute()
        pl_module.test_roc.compute()
        pl_module.test_prc.compute()
        pl_module.test_instance_prc.compute()
        pl_module.test_instance_cmx.compute()

        # Plot roc, prc and confusion matrix to disk and wandb
        fig_cmx, _ = pl_module.test_cmx.plot(cmap="Blues")
        fig_roc, _ = pl_module.test_roc.plot(score=True)
        fig_prc, _ = pl_module.test_prc.plot(score=True)
        fig_instance_cmx, _ = pl_module.test_instance_cmx.plot(cmap="Blues")
        fig_instance_prc, _ = pl_module.test_instance_prc.plot(score=True)

        # Check for a wandb or csv logger to log the images
        for pllogger in pl_module.loggers:
            if isinstance(pllogger, CSVLogger):
                fig_dir = Path(pllogger.log_dir) / "figures" / f"{self.test_set}-samples"
                fig_dir.mkdir(exist_ok=True, parents=True)
                fig_cmx.savefig(fig_dir / f"cmx_{pl_module.global_step}.png")
                fig_roc.savefig(fig_dir / f"roc_{pl_module.global_step}.png")
                fig_prc.savefig(fig_dir / f"prc_{pl_module.global_step}.png")
                fig_instance_cmx.savefig(fig_dir / f"instance_cmx_{pl_module.global_step}.png")
                fig_instance_prc.savefig(fig_dir / f"instance_prc_{pl_module.global_step}.png")
            if isinstance(pllogger, WandbLogger):
                wandb_run: Run = pllogger.experiment
                wandb_run.log({f"{self.test_set}/cmx": wandb.Image(fig_cmx)}, commit=False)
                wandb_run.log({f"{self.test_set}/roc": wandb.Image(fig_roc)}, commit=False)
                wandb_run.log({f"{self.test_set}/prc": wandb.Image(fig_prc)}, commit=False)
                wandb_run.log({f"{self.test_set}/instance_cmx": wandb.Image(fig_instance_cmx)}, commit=False)
                wandb_run.log({f"{self.test_set}/instance_prc": wandb.Image(fig_instance_prc)}, commit=False)

        fig_cmx.clear()
        fig_roc.clear()
        fig_prc.clear()
        fig_instance_cmx.clear()
        fig_instance_prc.clear()
        plt.close("all")

        # This will also commit the accumulated plots
        pl_module.log_dict(pl_module.test_metrics.compute())

        pl_module.test_metrics.reset()
        pl_module.test_roc.reset()
        pl_module.test_prc.reset()
        pl_module.test_cmx.reset()
        pl_module.test_instance_prc.reset()
        pl_module.test_instance_cmx.reset()
