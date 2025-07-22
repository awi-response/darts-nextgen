import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
import xarray as xr
from darts_acquisition import load_arcticdem, load_tcvis
from darts_ensemble import EnsembleV1
from darts_export import export_tile
from darts_postprocessing import prepare_export
from darts_preprocessing import preprocess_legacy_fast

logger = logging.getLogger(__name__)


@dataclass
class RayDataset:
    """A wrapper for xarray.Dataset to be used with Ray.

    This class is used to ensure that the dataset can be serialized and deserialized correctly.
    """

    dataset: xr.Dataset


@dataclass
class RayDataArray:
    """A wrapper for xarray.DataArray to be used with Ray.

    This class is used to ensure that the data array can be serialized and deserialized correctly.
    """

    data_array: xr.DataArray


class RayDataDict(TypedDict):
    tile: RayDataset | None
    adem: RayDataset | None
    tcvis: RayDataset | None
    tilekey: Any  # The key to identify the tile, e.g. a path or a tile id
    outpath: str  # The path to the output directory
    tile_id: str  # The id of the tile, e.g. the name of the file or the tile id


# @ray.remote(num_cpus=1, num_gpus=1)
class _RayEnsembleV1:
    def __init__(self, model_dict: Any):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble = EnsembleV1(model_dict=model_dict, device=device)

    def __call__(
        self,
        row: RayDataDict,
        *,
        patch_size: int,
        overlap: int,
        batch_size: int,
        reflection: int,
        write_model_outputs: bool,
    ) -> RayDataDict:
        tile = row["tile"].dataset
        tile = self.ensemble(
            tile,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            reflection=reflection,
            keep_inputs=write_model_outputs,
        )
        row["tile"] = RayDataset(tile)
        return row


# TODO: Das hier aufdröseln, damit loading und preprocessing separat läuft
# Dazu muss ein neues preprocess_legacy_fast geschrieben werden.
def _load_aux(
    row: RayDataDict,
    *,
    arcticdem_dir: Path,
    arcticdem_resolution: int,
    buffer: int,
    tcvis_dir: Path,
) -> RayDataDict:
    tile = row["tile"].dataset
    arcticdem = load_arcticdem(
        tile.odc.geobox,
        data_dir=arcticdem_dir,
        resolution=arcticdem_resolution,
        buffer=buffer,
    )
    tcvis = load_tcvis(tile.odc.geobox, tcvis_dir)
    row["adem"] = RayDataset(arcticdem)
    row["tcvis"] = RayDataset(tcvis)
    return row


def _preprocess_ray(
    row: RayDataDict,
    *,
    tpi_outer_radius: int,
    tpi_inner_radius: int,
    device: int | Literal["cuda", "cpu"],
):
    tile = row["tile"].dataset
    arcticdem = row["adem"].dataset
    tcvis = row["tcvis"].dataset
    tile = preprocess_legacy_fast(
        tile,
        arcticdem,
        tcvis,
        tpi_outer_radius,
        tpi_inner_radius,
        device,
    )
    row["tile"] = RayDataset(tile)
    row["adem"] = None
    row["tcvis"] = None
    return row


def _prepare_export_ray(
    row: RayDataDict,
    *,
    binarization_threshold: float,
    mask_erosion_size: int,
    min_object_size: int,
    quality_level: int,
    models: dict[str, Any],
    write_model_outputs: bool,
    device: int | Literal["cuda", "cpu"],
):
    tile = row["tile"].dataset
    tile = prepare_export(
        tile,
        bin_threshold=binarization_threshold,
        mask_erosion_size=mask_erosion_size,
        # TODO: edge_erosion_size
        min_object_size=min_object_size,
        quality_level=quality_level,
        ensemble_subsets=models.keys() if write_model_outputs else [],
        device=device,
    )
    row["tile"] = RayDataset(tile)
    return row


def _export_tile_ray(
    row: RayDataDict,
    *,
    export_bands: list[str],
    models: dict[str, Any],
    write_model_outputs: bool,
) -> RayDataDict:
    tile = row["tile"].dataset
    outpath = Path(row["outpath"])
    export_tile(
        tile,
        outpath,
        bands=export_bands,
        ensemble_subsets=models.keys() if write_model_outputs else [],
    )
    del row["tile"]

    tilekey = row["tilekey"]
    tile_id = row["tile_id"]
    logger.info(f"Processed sample '{tilekey}' ({tile_id=}).")
    return {
        "tilekey": tilekey,
        "tile_id": tile_id,
        "outpath": str(outpath),
    }
