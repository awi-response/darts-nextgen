"""Legacy pipeline for Sentinel 2 data with optimized preprocessing."""

from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from darts.legacy_pipeline._base import AquisitionData, _BasePipeline, _FastMixin, _S2Mixin


@dataclass
class LegacyNativeSentinel2PipelineFast(_FastMixin, _S2Mixin, _BasePipeline):
    """Pipeline for Sentinel 2 data with optimized preprocessing.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
            [italic magenta]Defaults to Path("data/input/sentinel2").[/italic magenta]
        output_data_dir (Path): The "output" directory. Defaults to Path("data/output").
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
            Defaults to Path("data/download/arcticdem").
        tcvis_dir (Path): The directory containing the TCVis data. Defaults to Path("data/download/tcvis").
        model_dir (Path): The path to the models to use for segmentation. Defaults to Path("models").
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
        device (Literal["cuda", "cpu"] | int, optional): The device to run the model on.
            If "cuda" take the first device (0), if int take the specified device.
            If "auto" try to automatically select a free GPU (<50% memory usage).
            Defaults to "cuda" if available, else "cpu".
        dask_worker (int, optional): The number of Dask workers to use. Defaults to min(16, mp.cpu_count() - 1).
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        ee_use_highvolume (bool, optional): Whether to use the high volume server (https://earthengine-highvolume.googleapis.com).
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in m. Defaults to 100m.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in m. Defaults to 0.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.
        reflection (int, optional): The reflection padding to use for inference. Defaults to 0.
        binarization_threshold (float, optional): The threshold to binarize the probabilities. Defaults to 0.5.
        mask_erosion_size (int, optional): The size of the disk to use for mask erosion and the edge-cropping.
            Defaults to 10.
        min_object_size (int, optional): The minimum object size to keep in pixel. Defaults to 32.
        use_quality_mask (bool, optional): Whether to use the "quality" mask instead of the "valid" mask
            to mask the output.
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.

    """

    def _get_data(self, fpath: Path):
        from darts_acquisition import load_arcticdem, load_s2_masks, load_s2_scene, load_tcvis

        optical = load_s2_scene(fpath)
        arcticdem = load_arcticdem(
            optical.odc.geobox, self.arcticdem_dir, resolution=10, buffer=ceil(self.tpi_outer_radius / 10 * sqrt(2))
        )
        tcvis = load_tcvis(optical.odc.geobox, self.tcvis_dir)
        data_masks = load_s2_masks(fpath, optical.odc.geobox)
        aqdata = AquisitionData(optical, arcticdem, tcvis, data_masks)
        return aqdata


def run_native_sentinel2_pipeline_fast(*, pipeline: Annotated[LegacyNativeSentinel2PipelineFast, Parameter("*")]):
    """Search for all Sentinel 2 scenes in the given directory and runs the segmentation pipeline on them.

    Loads the ArcticDEM from a datacube instead of VRT which is a lot faster and does not need manual preprocessing.

    Args:
        pipeline (_LegacyNativeSentinel2PipelineFast): The pipeline to run.

    """
    pipeline.run()
