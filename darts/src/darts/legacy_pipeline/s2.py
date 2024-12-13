"""Legacy pipeline for Sentinel 2 data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from darts.legacy_pipeline._base import AquisitionData, _BasePipeline, _S2Mixin, _VRTMixin


@dataclass
class LegacyNativeSentinel2Pipeline(_S2Mixin, _VRTMixin, _BasePipeline):
    """Pipeline for Sentinel 2 data.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes. Defaults to Path("data/input/sentinel2").
        output_data_dir (Path): The "output" directory. Defaults to Path("data/output").
        arcticdem_slope_vrt (Path): The path to the ArcticDEM slope VRT file.
            Defaults to Path("data/input/ArcticDEM/slope.vrt").
        arcticdem_elevation_vrt (Path): The path to the ArcticDEM elevation VRT file.
            Defaults to Path("data/input/ArcticDEM/elevation.vrt").
        tcvis_dir (Path): The directory containing the TCVis data. Defaults to Path("data/download/tcvis").
        model_dir (Path): The path to the models to use for segmentation. Defaults to Path("models").
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


def run_native_sentinel2_pipeline(*, pipeline: Annotated[LegacyNativeSentinel2Pipeline, Parameter("*")]):
    """Search for all Sentinel scenes in the given directory and runs the segmentation pipeline on them."""
    pipeline.run()
