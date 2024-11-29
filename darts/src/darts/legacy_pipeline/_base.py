import logging
import multiprocessing as mp
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

AquisitionData = namedtuple("AquisitionData", ["optical", "arcticdem", "tcvis", "data_masks"])


@dataclass
class _BasePipeline:
    """Base class for all pipelines.

    This class provides the run method which is the main entry point for all pipelines.

    This class is meant to be subclassed by the specific pipelines.
    These specific pipelines must implement the following methods:

    - "_path_generator" which generates the paths to the data (e.g. through Source Mixin)
    - "_get_data" which loads the data for a given path
    - "_preprocess" which preprocesses the data (e.g. through Processing Mixin)

    It is possible to implement these functions, by subclassing other mixins, e.g. _S2Mixin.

    The main class must be also a dataclass, to fully inherit all parameter of this class (and the mixins).
    """

    output_data_dir: Path
    tcvis_dir: Path
    model_dir: Path
    tcvis_model_name: str
    notcvis_model_name: str
    device: Literal["cuda", "cpu", "auto"] | int | None
    ee_project: str | None
    ee_use_highvolume: bool
    patch_size: int
    overlap: int
    batch_size: int
    reflection: int
    binarization_threshold: float
    mask_erosion_size: int
    min_object_size: int
    use_quality_mask: bool
    write_model_outputs: bool

    # These would be the type hints for the methods that need to be implemented
    # Leaving them uncommented would result in a NotImplementedError if Mixins are used
    # def _path_generator(self) -> Generator[tuple[Path, Path]]:
    #     raise NotImplementedError

    # def _get_data(self, fpath: Path) -> AquisitionData:
    #     raise NotImplementedError

    # def _preprocess(self, aqdata: AquisitionData) -> xr.Dataset:
    #     raise NotImplementedError

    def run(self):
        import torch
        from darts_ensemble.ensemble_v1 import EnsembleV1
        from darts_export.inference import InferenceResultWriter
        from darts_postprocessing import prepare_export
        from dask.distributed import Client, LocalCluster
        from odc.stac import configure_rio

        from darts.utils.cuda import debug_info, decide_device
        from darts.utils.earthengine import init_ee

        debug_info()
        self.device = decide_device(self.device)
        init_ee(self.ee_project, self.ee_use_highvolume)

        ensemble = EnsembleV1(
            self.model_dir / self.tcvis_model_name,
            self.model_dir / self.notcvis_model_name,
            device=torch.device(self.device),
        )

        # Init Dask stuff with a context manager
        with LocalCluster(n_workers=mp.cpu_count() - 1) as cluster, Client(cluster) as client:
            logger.info(f"Using Dask client: {client}")
            configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
            logger.info("Configured Rasterio with Dask")

            # Iterate over all the data (_path_generator)
            for fpath, outpath in self._path_generator():
                try:
                    aqdata = self._get_data(fpath)
                    tile = self._preprocess(aqdata)

                    tile = ensemble.segment_tile(
                        tile,
                        patch_size=self.patch_size,
                        overlap=self.overlap,
                        batch_size=self.batch_size,
                        reflection=self.reflection,
                        keep_inputs=self.write_model_outputs,
                    )
                    tile = prepare_export(
                        tile,
                        self.binarization_threshold,
                        self.mask_erosion_size,
                        self.min_object_size,
                        self.use_quality_mask,
                        self.device,
                    )

                    outpath.mkdir(parents=True, exist_ok=True)
                    writer = InferenceResultWriter(tile)
                    writer.export_probabilities(outpath)
                    writer.export_binarized(outpath)
                    writer.export_polygonized(outpath)
                except KeyboardInterrupt:
                    logger.warning("Keyboard interrupt detected.\nExiting...")
                    break
                except Exception as e:
                    logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
                    logger.exception(e)


# =============================================================================
# Processing mixins (they provide _preprocess method)
# =============================================================================
@dataclass
class _VRTMixin:
    arcticdem_slope_vrt: Path
    arcticdem_elevation_vrt: Path

    def _preprocess(self, aqdata: AquisitionData):
        from darts_preprocessing import preprocess_legacy

        return preprocess_legacy(aqdata.optical, aqdata.arcticdem, aqdata.tcvis, aqdata.data_masks)


@dataclass
class _FastMixin:
    arcticdem_dir: Path
    tpi_outer_radius: int
    tpi_inner_radius: int

    def _preprocess(self, aqdata: AquisitionData):
        from darts_preprocessing import preprocess_legacy_fast

        return preprocess_legacy_fast(
            aqdata.optical,
            aqdata.arcticdem,
            aqdata.tcvis,
            aqdata.data_masks,
            self.tpi_outer_radius,
            self.tpi_inner_radius,
            self.device,
        )


# =============================================================================
# Source mixins (they provide _path_generator method)
# =============================================================================
@dataclass
class _PlanetMixin:
    orthotiles_dir: Path
    scenes_dir: Path

    def _path_generator(self):
        # Find all PlanetScope orthotiles
        for fpath in self.orthotiles_dir.glob("*/*/"):
            tile_id = fpath.parent.name
            scene_id = fpath.name
            outpath = self.output_data_dir / tile_id / scene_id
            yield fpath, outpath

        # Find all PlanetScope scenes
        for fpath in self.scenes_dir.glob("*/"):
            scene_id = fpath.name
            outpath = self.output_data_dir / scene_id
            yield fpath, outpath


@dataclass
class _S2Mixin:
    sentinel2_dir: Path

    def _path_generator(self):
        for fpath in self.sentinel2_dir.glob("*/"):
            scene_id = fpath.name
            outpath = self.output_data_dir / scene_id
            yield fpath, outpath
