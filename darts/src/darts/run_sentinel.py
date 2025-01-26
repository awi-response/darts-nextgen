from darts.legacy_pipeline.s2_fast import LegacyNativeSentinel2PipelineFast
from darts.legacy_pipeline._base import AquisitionData, _BasePipeline, _FastMixin, _S2Mixin
from pathlib import Path
import multiprocessing as mp
from typing import Literal
from collections import namedtuple

if __name__ == "__main__":

    "read the input file"
    sentinel2_dir = Path("/taiga/toddn/pdg-files/ftp_shared_files/public/jokuep001/S2")
    orthotiles_dir = Path("/taiga/toddn/data/input/planet/PSOrthoTile")
    scenes_dir = Path("/taiga/toddn/data/input/planet/PSScene")
    output_data_dir = Path("/taiga/toddn/rts_dataset01/output")
    model_dir = Path("/taiga/toddn/rts_dataset01/models")
    tcvis_model_name = "/taiga/toddn/pdg-files/s2-tcvis-unetpp-resnext101-bestfromsweep_2025-01-15_fix.ckpt"
    notcvis_model_name = "RTS_v6_notcvis.pt"
    arcticdem_slope_vrt = Path("/taiga/toddn/rts_dataset01/input/ArcticDEM/slope.vrt")
    arcticdem_elevation_vrt = Path("/taiga/toddn/rts_dataset01/input/ArcticDEM/elevation.vrt")
    arcticdem_dir = Path("/taiga/toddn/rts_dataset01/input/ArcticDEM")
    tcvis_dir = Path("/home/toddn/darts-nextgen/tcvis_dir")

    print("Running the pipeline")
    print(f"Create the legacy pipeline ")
    tcvis_model_name: str = "RTS_v6_tcvis_s2native.pt"
    notcvis_model_name: str = "RTS_v6_notcvis_s2native.pt"
    device: Literal["cuda", "cpu", "auto"] | int | None = None
    dask_worker: int = min(16, mp.cpu_count() - 1)  # noqa: RUF009
    ee_project: str | None = None
    ee_use_highvolume: bool = True
    patch_size: int = 1024
    overlap: int = 256
    batch_size: int = 8
    reflection: int = 0
    binarization_threshold: float = 0.5
    mask_erosion_size: int = 10
    min_object_size: int = 32
    use_quality_mask: bool = False
    write_model_outputs: bool = False

    AquisitionData = namedtuple("AquisitionData", ["optical", "arcticdem", "tcvis", "data_masks"])


    fast_mixin = _FastMixin(arcticdem_dir = arcticdem_dir, tpi_outer_radius = 100, tpi_inner_radius = 0)

    s2_mixin = _S2Mixin(
        sentinel2_dir = Path(sentinel2_dir)
    )

    base_pipeline = _BasePipeline(
        output_data_dir = output_data_dir,
        tcvis_dir = tcvis_dir,
        model_dir = model_dir,
        tcvis_model_name = tcvis_model_name,
        notcvis_model_name = notcvis_model_name,
        device = device,
        dask_worker = dask_worker , # noqa: RUF009
        ee_project = ee_project,
        ee_use_highvolume = ee_use_highvolume,
        patch_size = patch_size,
        overlap = overlap,
        batch_size = batch_size,
        reflection = reflection,
        binarization_threshold = binarization_threshold,
        mask_erosion_size = mask_erosion_size,
        min_object_size = min_object_size,
        use_quality_mask = use_quality_mask,
        write_model_outputs = write_model_outputs,
    )

    pipeline = LegacyNativeSentinel2PipelineFast(
        output_data_dir=output_data_dir,
        tcvis_dir=tcvis_dir,
        model_dir=model_dir,
        tcvis_model_name=tcvis_model_name,
        notcvis_model_name=notcvis_model_name,
        device=device,
        dask_worker=dask_worker,  # noqa: RUF009
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
    )

    # pipeline = LegacyNativeSentinel2PipelineFast(
    #     _FastMixin = fast_mixin,
    #     _S2Mixin = s2_mixin,
    #     _BasePipeline = base_pipeline
    # )

    print('got pipeline')