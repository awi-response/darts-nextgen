import logging
import multiprocessing as mp
from collections import namedtuple
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

AquisitionData = namedtuple("AquisitionData", ["optical", "arcticdem", "tcvis", "data_masks"])


@dataclass
class _BasePipeline:
    """Base class for all legacy pipelines.

    This class provides the run method which is the main entry point for all pipelines.

    This class is meant to be subclassed by the specific pipelines.
    These specific pipelines must implement the following methods:

    - "_path_generator" which generates the paths to the data (e.g. through Source Mixin)
    - "_get_data" which loads the data for a given path
    - "_preprocess" which preprocesses the data (e.g. through Processing Mixin)

    It is possible to implement these functions, by subclassing other mixins, e.g. _S2Mixin.

    The main class must be also a dataclass, to fully inherit all parameter of this class (and the mixins).
    """

    output_data_dir: Path = Path("data/output")
    tcvis_dir: Path = Path("data/download/tcvis")
    model_dir: Path = Path("models")
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

    def __post_init__(self):
        from darts.utils.cuda import debug_info

        debug_info()

        from darts.utils.earthengine import init_ee

        init_ee(self.ee_project, self.ee_use_highvolume)

    def _path_generator(self) -> Generator[tuple[Path, Path]]:
        raise NotImplementedError

    def _get_data(self, fpath: Path) -> AquisitionData:
        raise NotImplementedError

    def _preprocess(self, aqdata: AquisitionData):
        raise NotImplementedError

    def run(self):
        import torch
        from darts_ensemble.ensemble_v1 import EnsembleV1
        from darts_export.inference import InferenceResultWriter
        from darts_postprocessing import prepare_export
        from dask.distributed import Client, LocalCluster
        from odc.stac import configure_rio

        from darts.utils.cuda import decide_device

        self.device = decide_device(self.device)

        ensemble = EnsembleV1(
            self.model_dir / self.tcvis_model_name,
            self.model_dir / self.notcvis_model_name,
            device=torch.device(self.device),
        )

        # Init Dask stuff with a context manager
        with LocalCluster(n_workers=self.dask_worker) as cluster, Client(cluster) as client:
            logger.info(f"Using Dask client: {client} on cluster {cluster}")
            logger.info(f"Dashboard available at: {client.dashboard_link}")
            configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
            logger.info("Configured Rasterio with Dask")

            # Iterate over all the data (_path_generator)
            n_tiles = 0
            paths = sorted(self._path_generator())
            logger.info(f"Found {len(paths)} tiles to process.")
            for i, (fpath, outpath) in enumerate(paths):
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
                    n_tiles += 1
                    logger.info(f"Processed sample {i + 1} of {len(paths)} '{fpath.resolve()}'.")
                except KeyboardInterrupt:
                    logger.warning("Keyboard interrupt detected.\nExiting...")
                    raise KeyboardInterrupt
                except Exception as e:
                    logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
                    logger.exception(e)
            else:
                logger.info(f"Processed {n_tiles} tiles to {self.output_data_dir.resolve()}.")


# =============================================================================
# Processing mixins (they provide _preprocess method)
# =============================================================================
@dataclass
class _VRTMixin:
    arcticdem_slope_vrt: Path = Path("data/input/ArcticDEM/slope.vrt")
    arcticdem_elevation_vrt: Path = Path("data/input/ArcticDEM/elevation.vrt")

    def _preprocess(self, aqdata: AquisitionData):
        from darts_preprocessing import preprocess_legacy

        return preprocess_legacy(aqdata.optical, aqdata.arcticdem, aqdata.tcvis, aqdata.data_masks)


@dataclass
class _FastMixin:
    arcticdem_dir: Path = Path("data/download/arcticdem")
    tpi_outer_radius: int = 100
    tpi_inner_radius: int = 0

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
    orthotiles_dir: Path = Path("data/input/planet/PSOrthoTile")
    scenes_dir: Path = Path("data/input/planet/PSScene")

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
    sentinel2_dir: Path = Path("data/input/sentinel2")

    def _path_generator(self):
        for fpath in self.sentinel2_dir.glob("*/"):
            scene_id = fpath.name
            outpath = self.output_data_dir / scene_id
            yield fpath, outpath
