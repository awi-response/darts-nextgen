"""Training module for DARTS."""

import logging
import multiprocessing as mp
from datetime import datetime
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

import toml

logger = logging.getLogger(__name__)


# We hardcode these because they depend on the preprocessing used
NORM_FACTORS = {
    "red": 1 / 3000,
    "green": 1 / 3000,
    "blue": 1 / 3000,
    "nir": 1 / 3000,
    "ndvi": 1 / 20000,
    "relative_elevation": 1 / 30000,
    "slope": 1 / 90,
    "tc_brightness": 1 / 255,
    "tc_greenness": 1 / 255,
    "tc_wetness": 1 / 255,
}


def preprocess_s2_train_data(
    *,
    sentinel2_dir: Path,
    train_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    bands: list[str],
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    include_allzero: bool = False,
    include_nan_edges: bool = True,
):
    """Preprocess Sentinel 2 data for training.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        train_data_dir (Path): The "output" directory where the tensors are written to.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
        bands (list[str]): The bands to be used for training. Must be present in the preprocessing.
        device (Literal["cuda", "cpu"] | int, optional): The device to run the model on.
            If "cuda" take the first device (0), if int take the specified device.
            If "auto" try to automatically select a free GPU (<50% memory usage).
            Defaults to "cuda" if available, else "cpu".
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        ee_use_highvolume (bool, optional): Whether to use the high volume server (https://earthengine-highvolume.googleapis.com).
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in m. Defaults to 100m.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in m. Defaults to 0.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        include_allzero (bool, optional): Whether to include patches where the labels are all zero. Defaults to False.
        include_nan_edges (bool, optional): Whether to include patches where the input data has nan values at the edges.
            Defaults to True.

    """
    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    import torch
    from darts_acquisition.arcticdem import load_arcticdem_tile
    from darts_acquisition.s2 import load_s2_masks, load_s2_scene
    from darts_acquisition.tcvis import load_tcvis
    from darts_preprocessing import preprocess_legacy_fast
    from darts_segmentation.training.prepare_training import create_training_patches
    from dask.distributed import Client, LocalCluster
    from odc.stac import configure_rio

    from darts.utils.cuda import debug_info, decide_device
    from darts.utils.earthengine import init_ee

    debug_info()
    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)

    cluster = LocalCluster(n_workers=mp.cpu_count() - 1)
    logger.info(f"Created Dask cluster: {cluster}")
    client = Client(cluster)
    logger.info(f"Using Dask client: {client}")
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
    logger.info("Configured Rasterio with Dask")

    outpath_x = train_data_dir / "x"
    outpath_y = train_data_dir / "y"

    outpath_x.mkdir(exist_ok=True, parents=True)
    outpath_y.mkdir(exist_ok=True, parents=True)

    # We hardcode these because they depend on the preprocessing used
    norm_factors = {
        "red": 1 / 3000,
        "green": 1 / 3000,
        "blue": 1 / 3000,
        "nir": 1 / 3000,
        "ndvi": 1 / 20000,
        "relative_elevation": 1 / 30000,
        "slope": 1 / 90,
        "tc_brightness": 1 / 255,
        "tc_greenness": 1 / 255,
        "tc_wetness": 1 / 255,
    }
    # Filter out bands that are not in the specified bands
    norm_factors = {k: v for k, v in norm_factors.items() if k in bands}

    # Find all Sentinel 2 scenes
    n_patches = 0
    for fpath in sentinel2_dir.glob("*/"):
        try:
            optical = load_s2_scene(fpath)
            arcticdem = load_arcticdem_tile(
                optical.odc.geobox, arcticdem_dir, resolution=10, buffer=ceil(tpi_outer_radius / 10 * sqrt(2))
            )
            tcvis = load_tcvis(optical.odc.geobox, tcvis_dir)
            data_masks = load_s2_masks(fpath, optical.odc.geobox)

            tile = preprocess_legacy_fast(
                optical,
                arcticdem,
                tcvis,
                data_masks,
                tpi_outer_radius,
                tpi_inner_radius,
                device,
            )

            labels = gpd.read_file(fpath / f"{optical.attrs['s2_tile_id']}.shp")

            # Save the patches
            tile_id = optical.attrs["tile_id"]
            gen = create_training_patches(
                tile,
                labels,
                bands,
                norm_factors,
                patch_size,
                overlap,
                include_allzero,
                include_nan_edges,
            )
            for patch_id, (x, y) in enumerate(gen):
                torch.save(x, outpath_x / f"{tile_id}_pid{patch_id}.pt")
                torch.save(y, outpath_y / f"{tile_id}_pid{patch_id}.pt")
                n_patches += 1
            logger.info(f"Processed {tile_id} with {patch_id} patches.")

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            break

        except Exception as e:
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)

    # Save a config file as toml
    config = {
        "darts": {
            "sentinel2_dir": sentinel2_dir,
            "train_data_dir": train_data_dir,
            "arcticdem_dir": arcticdem_dir,
            "tcvis_dir": tcvis_dir,
            "bands": bands,
            "norm_factors": norm_factors,
            "device": device,
            "ee_project": ee_project,
            "ee_use_highvolume": ee_use_highvolume,
            "tpi_outer_radius": tpi_outer_radius,
            "tpi_inner_radius": tpi_inner_radius,
            "patch_size": patch_size,
            "overlap": overlap,
            "include_allzero": include_allzero,
            "include_nan_edges": include_nan_edges,
            "n_patches": n_patches,
        }
    }
    with open(train_data_dir / "config.toml", "w") as f:
        toml.dump(config, f)

    logger.info(f"Saved {n_patches} patches to {train_data_dir}")

    client.close()
    cluster.close()


def train_smp(
    *,
    train_data_dir: Path,
    artifact_dir: Path = Path("lightning_logs"),
    model_arch: str = "Unet",
    model_encoder: str = "dpn107",
    model_encoder_weights: str | None = None,
    augment: bool = True,
    batch_size: int = 8,
    num_workers: int = 0,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    run_name: str | None = None,
):
    """Run the training of the SMP model.

    Please see https://smp.readthedocs.io/en/latest/index.html for model configurations.

    Args:
        train_data_dir (Path): Path to the training data directory.
        artifact_dir (Path, optional): Path to the training output directory.
            Will contain checkpoints and metrics. Defaults to Path("lightning_logs").
        model_arch (str, optional): Model architecture to use. Defaults to "Unet".
        model_encoder (str, optional): Encoder to use. Defaults to "dpn107".
        model_encoder_weights (str | None, optional): Path to the encoder weights. Defaults to None.
        augment (bool, optional): Weather to apply augments or not. Defaults to True.
        batch_size (int, optional): Batch Size. Defaults to 8.
        num_workers (int, optional): Number of Dataloader workers. Defaults to 0.
        wandb_entity (str | None, optional): Weights and Biases Entity. Defaults to None.
        wandb_project (str | None, optional): Weights and Biases Project. Defaults to None.
        run_name (str | None, optional): Name of this run, as a further grouping method for logs etc. Defaults to None.

    """
    import lightning as L  # noqa: N812
    import lovely_tensors
    import torch
    from darts_segmentation.segment import SMPSegmenterConfig
    from darts_segmentation.training.data import DartsDataModule
    from darts_segmentation.training.module import SMPSegmenter
    from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    lovely_tensors.monkey_patch()

    torch.set_float32_matmul_precision("medium")

    preprocess_config = toml.load(train_data_dir / "config.toml")["darts"]

    config = SMPSegmenterConfig(
        input_combination=preprocess_config["bands"],
        model={
            "arch": model_arch,
            "encoder_name": model_encoder,
            "encoder_weights": model_encoder_weights,
            "in_channels": len(preprocess_config["bands"]),
            "classes": 1,
        },
        norm_factors=preprocess_config["norm_factors"],
    )

    datamodule = DartsDataModule(train_data_dir, batch_size, augment, num_workers)

    model = SMPSegmenter(config)

    trainer_loggers = [
        CSVLogger(save_dir=artifact_dir, name=run_name),
    ]
    if wandb_entity and wandb_project:
        wandb_logger = WandbLogger(save_dir=artifact_dir, name=run_name, project=wandb_project, entity=wandb_entity)
        trainer_loggers.append(wandb_logger)
    early_stopping = EarlyStopping(monitor="val/JaccardIndex", mode="max", patience=5)
    callbacks = [
        early_stopping,
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=trainer_loggers,
        check_val_every_n_epoch=3,
    )
    trainer.fit(model, datamodule)

    # TODO: save with own config etc.
    # Add timestamp


def convert_lightning_checkpoint(
    *,
    lightning_checkpoint: Path,
    out_directory: Path,
    checkpoint_name: str,
    framework: str = "smp",
):
    """Convert a lightning checkpoint to our own format.

    The final checkpoint will contain the model configuration and the state dict.
    It will be saved to:

    ```python
        out_directory / f"{checkpoint_name}_{formatted_date}.ckpt"
    ```

    Args:
        lightning_checkpoint (Path): Path to the lightning checkpoint.
        out_directory (Path): Output directory for the converted checkpoint.
        checkpoint_name (str): A unique name of the new checkpoint.
        framework (str, optional): The framework used for the model. Defaults to "smp".

    """
    import torch

    logger.debug(f"Loading checkpoint from {lightning_checkpoint.resolve()}")
    lckpt = torch.load(lightning_checkpoint, weights_only=False)

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    config = lckpt["hyper_parameters"]
    config["time"] = formatted_date
    config["name"] = checkpoint_name
    config["model_framework"] = framework

    own_ckpt = {
        "config": config,
        "statedict": lckpt["state_dict"],
    }

    out_directory.mkdir(exist_ok=True, parents=True)

    out_checkpoint = out_directory / f"{checkpoint_name}_{formatted_date}.ckpt"

    torch.save(own_ckpt, out_checkpoint)

    logger.info(f"Saved converted checkpoint to {out_checkpoint.resolve()}")
