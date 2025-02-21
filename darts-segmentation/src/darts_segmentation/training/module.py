"""Training script for DARTS segmentation."""

import lightning as L  # noqa: N812
import segmentation_models_pytorch as smp
import torch.optim as optim

from darts_segmentation.segment import SMPSegmenterConfig


class SMPSegmenter(L.LightningModule):
    """Lightning module for training a segmentation model using the segmentation_models_pytorch library."""

    def __init__(
        self,
        config: SMPSegmenterConfig,
        learning_rate: float = 1e-5,
        gamma: float = 0.9,
        focal_loss_alpha: float | None = None,
        focal_loss_gamma: float = 2.0,
        val_set: str = "val",
        test_set: str = "test",
        **kwargs,
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
            val_set (str, optional): Name of the validation set. Only used for naming the validation metrics.
                Defaults to "val".
            test_set (str, optional): Name of the test set. Only used for naming the test metrics. Defaults to "test".
            kwargs: Additional keyword arguments which should be saved to the hyperparameter file.

        """
        super().__init__()

        # This saves config, learning_rate and gamma under self.hparams
        self.save_hyperparameters(ignore=["test_set", "val_set"])
        self.model = smp.create_model(**config["model"], activation="sigmoid")

        # Assumes that the training preparation was done with setting invalid pixels in the mask to 2
        self.loss_fn = smp.losses.FocalLoss(
            mode="binary", alpha=focal_loss_alpha, gamma=focal_loss_gamma, ignore_index=2
        )

        assert "/" not in val_set, "val_set must not contain '/'"
        assert "/" not in test_set, "test_set must not contain '/'"
        self.val_set = val_set
        self.test_set = test_set

    def __repr__(self):  # noqa: D105
        return f"SMPSegmenter({self.hparams['config']['model']})"

    def training_step(self, batch, batch_idx):  # noqa: D102
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        loss = self.loss_fn(y_hat, y.long())
        self.log("train/loss", loss)
        return {
            "loss": loss,
            "y_hat": y_hat,
        }

    def on_train_epoch_end(self):  # noqa: D102
        self.log("learning_rate", self.lr_schedulers().get_last_lr()[0])

    def validation_step(self, batch, batch_idx):  # noqa: D102
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        loss = self.loss_fn(y_hat, y.long())
        self.log(f"{self.val_set}/loss", loss)

        return {
            "loss": loss,
            "y_hat": y_hat,
        }

    def test_step(self, batch, batch_idx):  # noqa: D102
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        loss = self.loss_fn(y_hat, y.long())
        self.log(f"{self.test_set}/loss", loss)

        return {
            "loss": loss,
            "y_hat": y_hat,
        }

    def configure_optimizers(self):  # noqa: D102
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
