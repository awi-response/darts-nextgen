"""Legacy pipeline for Sentinel 2 data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from darts.legacy_pipeline._base import AquisitionData, _BasePipeline, _S2Mixin, _VRTMixin


@dataclass
class _LegacyNativeSentinel2Pipeline(_BasePipeline, _S2Mixin, _VRTMixin):
    def _get_data(self, fpath: Path):
        from darts_acquisition.arcticdem import load_arcticdem_from_vrt
        from darts_acquisition.s2 import load_s2_masks, load_s2_scene
        from darts_acquisition.tcvis import load_tcvis

        optical = load_s2_scene(fpath)
        arcticdem = load_arcticdem_from_vrt(self.arcticdem_slope_vrt, self.arcticdem_elevation_vrt, optical)
        tcvis = load_tcvis(optical.odc.geobox, self.tcvis_dir)
        data_masks = load_s2_masks(fpath, optical.odc.geobox)
        aqdata = AquisitionData(optical, arcticdem, tcvis, data_masks)
        return aqdata


def run_native_sentinel2_pipeline(
    *,
    sentinel2_dir: Path,
    output_data_dir: Path,
    arcticdem_slope_vrt: Path,
    arcticdem_elevation_vrt: Path,
    tcvis_dir: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis_s2native.pt",
    notcvis_model_name: str = "RTS_v6_notcvis_s2native.pt",
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
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
    """Search for all Sentinel scenes in the given directory and runs the segmentation pipeline on them.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        output_data_dir (Path): The "output" directory.
        arcticdem_slope_vrt (Path): The path to the ArcticDEM slope VRT file.
        arcticdem_elevation_vrt (Path): The path to the ArcticDEM elevation VRT file.
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

    Examples:
        Data directory structure:

        ```sh
            data/input
            ├── ArcticDEM
            │   ├── elevation.vrt
            │   ├── slope.vrt
            │   ├── relative_elevation
            │   │   └── 4372514_relative_elevation_100.tif
            │   └── slope
            │       └── 4372514_slope.tif
            └── sentinel2
                └── 20220826T200911_20220826T200905_T17XMJ/
                    ├── 20220826T200911_20220826T200905_T17XMJ_SCL_clip.tif
                    └── 20220826T200911_20220826T200905_T17XMJ_SR_clip.tif
        ```

        then the config should be

        ```
        ...
        sentinel2_dir: data/input/sentinel2
        arcticdem_slope_vrt: data/input/ArcticDEM/slope.vrt
        arcticdem_elevation_vrt: data/input/ArcticDEM/elevation.vrt
        ```


    """
    _LegacyNativeSentinel2Pipeline(
        sentinel2_dir=sentinel2_dir,
        output_data_dir=output_data_dir,
        arcticdem_elevation_vrt=arcticdem_elevation_vrt,
        arcticdem_slope_vrt=arcticdem_slope_vrt,
        tcvis_dir=tcvis_dir,
        model_dir=model_dir,
        tcvis_model_name=tcvis_model_name,
        notcvis_model_name=notcvis_model_name,
        device=device,
        ee_project=ee_project,
        ee_use_highvolume=ee_use_highvolume,
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
