"""Legacy pipeline for Planet data with optimized preprocessing."""

from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

from darts.legacy_pipeline._base import AquisitionData, _BasePipeline, _FastMixin, _PlanetMixin


@dataclass
class _LegacyNativePlanetPipelineFast(_BasePipeline, _PlanetMixin, _FastMixin):
    def _get_data(self, fpath: Path):
        from darts_acquisition.arcticdem import load_arcticdem_tile
        from darts_acquisition.planet import load_planet_masks, load_planet_scene
        from darts_acquisition.tcvis import load_tcvis

        optical = load_planet_scene(fpath)
        arcticdem = load_arcticdem_tile(
            optical.odc.geobox, self.arcticdem_dir, resolution=10, buffer=ceil(self.tpi_outer_radius / 10 * sqrt(2))
        )
        tcvis = load_tcvis(optical.odc.geobox, self.tcvis_dir)
        data_masks = load_planet_masks(fpath)
        aqdata = AquisitionData(optical, arcticdem, tcvis, data_masks)
        return aqdata


def run_native_planet_pipeline_fast(
    *,
    orthotiles_dir: Path,
    scenes_dir: Path,
    output_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis.pt",
    notcvis_model_name: str = "RTS_v6_notcvis.pt",
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
    reflection: int = 0,
    binarization_threshold: float = 0.5,
    mask_erosion_size: int = 10,
    min_object_size: int = 32,
    use_quality_mask: bool = False,
    write_model_outputs: bool = False,
):
    """Search for all PlanetScope scenes in the given directory and runs the segmentation pipeline on them.

    Loads the ArcticDEM from a datacube instead of VRT which is a lot faster and does not need manual preprocessing.

    Args:
        orthotiles_dir (Path): The directory containing the PlanetScope orthotiles.
        scenes_dir (Path): The directory containing the PlanetScope scenes.
        output_data_dir (Path): The "output" directory.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
        model_dir (Path): The path to the models to use for segmentation.
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
        device (Literal["cuda", "cpu"] | int, optional): The device to run the model on.
            If "cuda" take the first device (0), if int take the specified device.
            If "auto" try to automatically select a free GPU (<50% memory usage).
            Defaults to "cuda" if available, else "cpu".
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        ee_use_highvolume (bool, optional): Whether to use the high volume server (https://earthengine-highvolume.googleapis.com).
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in m. Defaults 100m.
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
    _LegacyNativePlanetPipelineFast(
        orthotiles_dir=orthotiles_dir,
        scenes_dir=scenes_dir,
        output_data_dir=output_data_dir,
        arcticdem_dir=arcticdem_dir,
        tcvis_dir=tcvis_dir,
        model_dir=model_dir,
        tcvis_model_name=tcvis_model_name,
        notcvis_model_name=notcvis_model_name,
        device=device,
        ee_project=ee_project,
        ee_use_highvolume=ee_use_highvolume,
        tpi_outer_radius=tpi_outer_radius,
        tpi_inner_radius=tpi_inner_radius,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        reflection=reflection,
        binarization_threshold=binarization_threshold,
        mask_erosion_size=mask_erosion_size,
        min_object_size=min_object_size,
        use_quality_mask=use_quality_mask,
        write_model_outputs=write_model_outputs,
    ).run()
