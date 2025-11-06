"""Ray implementation of the v2 pipelines."""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import cached_property
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import toml
from cyclopts import Parameter
from darts_utils.paths import DefaultPaths, paths

if TYPE_CHECKING:
    from darts.pipelines._ray_wrapper import RayDataDict

logger = logging.getLogger(__name__)


class RayInputDict(TypedDict):
    """A dictionary to hold the input data for Ray tasks.

    This is used to ensure that the input data can be serialized and deserialized correctly.
    """

    tilekey: Any  # The key to identify the tile, e.g. a path or a tile id
    outpath: str  # The path to the output directory
    tile_id: str  # The id of the tile, e.g. the name of the file or the tile id


@Parameter(name="*")
@dataclass
class _BaseRayPipeline(ABC):
    """Base class for all v2 pipelines.

    This class provides the run method which is the main entry point for all pipelines.

    This class is meant to be subclassed by the specific pipelines.
    These pipeliens must implement the _aqdata_generator method.

    The main class must be also a dataclass, to fully inherit all parameter of this class (and the mixins).

    Args:
        - num_cpus (int): The number of CPUs to use for the Ray tasks. Defaults to 1.

    """

    model_files: list[Path] = None
    default_dirs: DefaultPaths = field(default_factory=lambda: DefaultPaths())
    output_data_dir: Path | None = None
    arcticdem_dir: Path | None = None
    tcvis_dir: Path | None = None
    num_cpus: int = 1
    devices: list[int] | None = None
    device: Literal["cuda", "cpu", "auto"] | int | None = None
    ee_project: str | None = None
    ee_use_highvolume: bool = True
    tpi_outer_radius: int = 100
    tpi_inner_radius: int = 0
    patch_size: int = 1024
    overlap: int = 256
    batch_size: int = 8
    reflection: int = 0
    binarization_threshold: float = 0.5
    mask_erosion_size: int = 10
    edge_erosion_size: int | None = None
    min_object_size: int = 32
    quality_level: int | Literal["high_quality", "low_quality", "none"] = 1
    export_bands: list[str] = field(
        default_factory=lambda: ["probabilities", "binarized", "polygonized", "extent", "thumbnail"]
    )
    write_model_outputs: bool = False
    overwrite: bool = False
    offline: bool = False
    debug_data: bool = False

    def __post_init__(self):
        paths.set_defaults(self.default_dirs)
        self.output_data_dir = self.output_data_dir or paths.out
        self.model_files = self.model_files or list(paths.models.glob("*.pt"))
        if self.arcticdem_dir is None:
            arcticdem_resolution = self._arcticdem_resolution()
            self.arcticdem_dir = paths.arcticdem(arcticdem_resolution)
        self.tcvis_dir = self.tcvis_dir or paths.tcvis()
        if self.edge_erosion_size is None:
            self.edge_erosion_size = self.mask_erosion_size

    @abstractmethod
    def _arcticdem_resolution(self) -> Literal[2, 10, 32]:
        """Return the resolution of the ArcticDEM data."""
        pass

    @abstractmethod
    def _get_tile_id(self, tilekey: Any) -> str:
        pass

    @abstractmethod
    def _tileinfos(self) -> list[tuple[Any, Path]]:
        # Yields a tuple
        # str: anything which id needed to load the tile, e.g. a path or a tile id
        # Path: the path to the output directory
        pass

    @abstractmethod
    def _load_tile(self, tileinfo: RayInputDict) -> "RayDataDict":
        pass

    def _validate(self):
        if self.model_files is None or len(self.model_files) == 0:
            raise ValueError("No model files provided. Please provide a list of model files.")
        if len(self.export_bands) == 0:
            raise ValueError("No export bands provided. Please provide a list of export bands.")

    def _dump_config(self) -> str:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"Starting pipeline at {current_time}.")

        # Storing the configuration as TOML file
        self.output_data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_data_dir / f"{current_time}.config.toml", "w") as f:
            config = asdict(self)
            for key, value in config.items():
                if isinstance(value, Path):
                    config[key] = str(value.resolve())
                elif isinstance(value, list):
                    config[key] = [str(v.resolve()) if isinstance(v, Path) else v for v in value]
                elif is_dataclass(value):
                    config[key] = asdict(value)
            toml.dump(config, f)
        return current_time

    def _create_auxiliary_datacubes(self, arcticdem: bool = True, tcvis: bool = True):
        import smart_geocubes

        from darts.utils.logging import LoggingManager

        LoggingManager.apply_logging_handlers("smart_geocubes")
        if arcticdem:
            arcticdem_resolution = self._arcticdem_resolution()
            if arcticdem_resolution == 2:
                accessor = smart_geocubes.ArcticDEM2m(self.arcticdem_dir)
            elif arcticdem_resolution == 10:
                accessor = smart_geocubes.ArcticDEM10m(self.arcticdem_dir)
            if not accessor.created:
                accessor.create(overwrite=False)
        if tcvis:
            accessor = smart_geocubes.TCTrend(self.tcvis_dir)
            if not accessor.created:
                accessor.create(overwrite=False)

    def _load_ensemble(self):
        import torch
        from darts_ensemble import EnsembleV1

        if isinstance(self.model_files, Path):
            self.model_files = [self.model_files]
        if len(self.model_files) == 1:
            self.write_model_outputs = False
        models = {model_file.stem: model_file for model_file in self.model_files}
        ensemble = EnsembleV1(models, device=torch.device(self.device))
        return ensemble, models

    def _check_aux_needs(self, ensemble) -> tuple[bool, bool]:
        required_bands = ensemble.required_bands
        arcticdem_bands = {"dem", "relative_elevation", "slope", "aspect", "hillshade", "curvature"}
        tcvis_bands = {"tc_brightness", "tc_greenness", "tc_wetness"}
        needs_arcticdem = len(required_bands.intersection(arcticdem_bands)) > 0
        needs_tcvis = len(required_bands.intersection(tcvis_bands)) > 0
        return needs_arcticdem, needs_tcvis

    def _result_metadata(self, tilekey: Any) -> dict:
        export_metadata = {
            "tileid": self._get_tile_id(tilekey),
            "modelfiles": [f.name for f in self.model_files],
            "tpiouter": self.tpi_outer_radius,
            "tpiinner": self.tpi_inner_radius,
            "patchsize": self.patch_size,
            "overlap": self.overlap,
            "reflection": self.reflection,
            "binarizethreshold": self.binarization_threshold,
            "maskerosion": self.mask_erosion_size,
            "edgeerosion": self.edge_erosion_size,
            "mmu": self.min_object_size,
            "qualitymask": self.quality_level,
        }
        return {f"DARTS_{k}": v for k, v in export_metadata.items()}

    def run(self):  # noqa: C901
        self._validate()
        current_time = self._dump_config()

        if self.devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self.devices)
        from darts.utils.cuda import debug_info

        debug_info()

        from darts.utils.cuda import decide_device
        from darts.utils.earthengine import init_ee

        self.device = decide_device(self.device)
        if not self.offline:
            init_ee(self.ee_project, self.ee_use_highvolume)

        import ray

        if os.getenv("RAY_ADDRESS"):
            # Connect to an existing Ray cluster (e.g., on GKE)
            ray_context = ray.init(address=os.getenv("RAY_ADDRESS"))
            logger.info(f"Connected to Ray cluster at {os.getenv('RAY_ADDRESS')}")
        else:
            # Fallback to local
            ray_context = ray.init(
                num_cpus=self.num_cpus,
                num_gpus=len(self.devices) if self.devices is not None else None,
            )
            logger.info("Started local Ray instance for debugging.")

        logger.debug(f"Ray initialized with context: {ray_context}")
        logger.info(f"Ray Dashboard URL: {ray_context.dashboard_url}")
        logger.debug(f"Ray cluster resources: {ray.cluster_resources()}")
        logger.debug(f"Ray available resources: {ray.available_resources()}")

        # Initlize ee in every worker
        @ray.remote
        def init_worker():
            init_ee(self.ee_project, self.ee_use_highvolume)

        if not self.offline:
            num_workers = int(ray.cluster_resources().get("CPU", 1))
            logger.info(f"Initializing {num_workers} Ray workers with Earth Engine.")
            ray.get([init_worker.remote() for _ in range(num_workers)])

        from darts_export import missing_outputs

        from darts.pipelines._ray_wrapper import (
            _export_tile_ray,
            _load_aux,
            _prepare_export_ray,
            _preprocess_ray,
            _RayEnsembleV1,
        )
        from darts.utils.logging import LoggingManager

        # Create the datacubes if they do not exist (empty containers that get filled over time)
        self._create_auxiliary_datacubes()

        # determine models to use
        ensemble, models = self._load_ensemble()
        arcticdem_resolution = self._arcticdem_resolution()
        adem_buffer = ceil(self.tpi_outer_radius / arcticdem_resolution * sqrt(2))

        # Get files to process
        needs_arcticdem, needs_tcvis = self._check_aux_needs(ensemble)
        tileinfo: list[RayInputDict] = []
        for i, (tilekey, outpath) in enumerate(self._tileinfos()):
            tile_id = self._get_tile_id(tilekey)
            if not self.overwrite:
                mo = missing_outputs(outpath, bands=self.export_bands, ensemble_subsets=ensemble.model_names)
                if mo == "none":
                    logger.info(f"Tile {tile_id} already processed. Skipping...")
                    continue
                if mo == "some":
                    logger.warning(
                        f"Tile {tile_id} seems to be already processed, but some outputs are missing. "
                        "Skipping because overwrite=False..."
                    )
                    continue
            tileinfo.append({
                "tilekey": tilekey,
                "outpath": str(outpath.resolve()),
                "tile_id": tile_id,
                "metadata": self._result_metadata(tilekey),
                "debug_data": self.debug_data,
            })
        logger.info(f"Found {len(tileinfo)} tiles to process.")

        # Ray data pipeline
        # TODO: setup device stuff correctly
        ds = ray.data.from_items(tileinfo)
        ds = ds.map(self._load_tile, num_cpus=1)
        if needs_arcticdem or needs_tcvis:
            ds = ds.map(
                _load_aux,
                fn_kwargs={
                    "arcticdem_dir": self.arcticdem_dir,
                    "arcticdem_resolution": arcticdem_resolution,
                    "buffer": adem_buffer,
                    "tcvis_dir": self.tcvis_dir,
                    "offline": self.offline,
                },
                num_cpus=1,
            )
        ds = ds.map(
            _preprocess_ray,
            fn_kwargs={
                "tpi_outer_radius": self.tpi_outer_radius,
                "tpi_inner_radius": self.tpi_inner_radius,
                "device": "cuda" if self.device == "cuda" or isinstance(self.device, int) else "cpu",
            },
            num_cpus=1,
            num_gpus=0.1,
            concurrency=4,
        )
        ds = ds.map(
            _RayEnsembleV1,
            fn_constructor_kwargs={"model_dict": models},
            fn_kwargs={
                "patch_size": self.patch_size,
                "overlap": self.overlap,
                "batch_size": self.batch_size,
                "reflection": self.reflection,
                "write_model_outputs": self.write_model_outputs,
            },
            num_cpus=1,
            num_gpus=0.8,
            concurrency=1,
        )
        ds = ds.map(
            _prepare_export_ray,
            fn_kwargs={
                "binarization_threshold": self.binarization_threshold,
                "mask_erosion_size": self.mask_erosion_size,
                "min_object_size": self.min_object_size,
                "quality_level": self.quality_level,
                "models": models,
                "write_model_outputs": self.write_model_outputs,
                "device": "cuda" if self.device == "cuda" or isinstance(self.device, int) else "cpu",
                "edge_erosion_size": self.edge_erosion_size,
            },
            num_cpus=1,
            num_gpus=0.1,
        )
        ds = ds.map(
            _export_tile_ray,
            fn_kwargs={
                "export_bands": self.export_bands,
                "models": models,
                "write_model_outputs": self.write_model_outputs,
            },
            num_cpus=1,
        )
        logger.debug(f"Ray dataset: {ds}")
        logger.info("Ray pipeline created. Starting execution...")
        # This should trigger the execution
        ds.write_parquet(f"local://{self.output_data_dir.resolve()!s}/ray_output.parquet")
        logger.info(f"Ray pipeline finished. Output written to {self.output_data_dir.resolve()!s}/ray_output.parquet")


# =============================================================================
# Source Pipeliens
# =============================================================================
@dataclass
class PlanetRayPipeline(_BaseRayPipeline):
    """Pipeline for PlanetScope data.

    Args:
        orthotiles_dir (Path): The directory containing the PlanetScope orthotiles.
        scenes_dir (Path): The directory containing the PlanetScope scenes.
        image_ids (list): The list of image ids to process. If None, all images in the directory will be processed.


        model_files (Path | list[Path]): The path to the models to use for segmentation.
            Can also be a single Path to only use one model. This implies `write_model_outputs=False`
            If a list is provided, will use an ensemble of the models.
        output_data_dir (Path): The "output" directory. Defaults to Path("data/output").
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
            Defaults to Path("data/download/arcticdem").
        tcvis_dir (Path): The directory containing the TCVis data. Defaults to Path("data/download/tcvis").
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
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.
        reflection (int, optional): The reflection padding to use for inference. Defaults to 0.
        binarization_threshold (float, optional): The threshold to binarize the probabilities. Defaults to 0.5.
        mask_erosion_size (int, optional): The size of the disk to use for mask erosion and the edge-cropping.
            Defaults to 10.
        min_object_size (int, optional): The minimum object size to keep in pixel. Defaults to 32.
        quality_level (int | Literal["high_quality", "low_quality", "none"], optional):
            The quality level to use for the segmentation. Can also be an int.
            In this case 0="none" 1="low_quality" 2="high_quality". Defaults to 1.
        export_bands (list[str], optional): The bands to export.
            Can be a list of "probabilities", "binarized", "polygonized", "extent", "thumbnail", "optical", "dem",
            "tcvis" or concrete band-names.
            Defaults to ["probabilities", "binarized", "polygonized", "extent", "thumbnail"].
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.

    """

    orthotiles_dir: Path = Path("data/input/planet/PSOrthoTile")
    scenes_dir: Path = Path("data/input/planet/PSScene")
    image_ids: list = None

    def _arcticdem_resolution(self) -> Literal[2]:
        return 2

    def _get_tile_id(self, tilekey: Path) -> str:
        from darts_acquisition import parse_planet_type

        try:
            fpath = tilekey
            planet_type = parse_planet_type(fpath)
            tile_id = fpath.parent.stem if planet_type == "orthotile" else fpath.stem
            return tile_id
        except Exception as e:
            logger.error("Could not parse Planet tile-id. Please check the input data.")
            logger.exception(e)
            raise e

    def _tileinfos(self) -> list[tuple[Path, Path]]:
        out = []
        # Find all PlanetScope orthotiles
        for fpath in self.orthotiles_dir.glob("*/*/"):
            tile_id = fpath.parent.name
            scene_id = fpath.name
            if self.image_ids is not None:
                if scene_id not in self.image_ids:
                    continue
            outpath = self.output_data_dir / tile_id / scene_id
            out.append((fpath.resolve(), outpath))

        # Find all PlanetScope scenes
        for fpath in self.scenes_dir.glob("*/"):
            scene_id = fpath.name
            if self.image_ids is not None:
                if scene_id not in self.image_ids:
                    continue
            outpath = self.output_data_dir / scene_id
            out.append((fpath.resolve(), outpath))
        out.sort()
        return out

    def _load_tile(self, tileinfo: RayInputDict) -> "RayDataDict":
        import xarray as xr
        from darts_acquisition import load_planet_masks, load_planet_scene

        from darts.pipelines._ray_wrapper import RayDataset

        fpath: Path = tileinfo["tilekey"]
        optical = load_planet_scene(fpath)
        data_masks = load_planet_masks(fpath)
        tile = xr.merge([optical, data_masks])
        tile = RayDataset(dataset=tile)
        return {"tile": tile, **tileinfo}

    @staticmethod
    def cli(*, pipeline: "PlanetRayPipeline"):
        """Run the sequential pipeline for Planet data."""
        pipeline.run()


# TODO: Apply changes from sequential
@dataclass
class Sentinel2RayPipeline(_BaseRayPipeline):
    """Pipeline for Sentinel 2 data based on an area of interest.

    Args:
        aoi_shapefile (Path): The shapefile containing the area of interest.
        start_date (str): The start date of the time series in YYYY-MM-DD format.
        end_date (str): The end date of the time series in YYYY-MM-DD format.
        max_cloud_cover (int): The maximum cloud cover percentage to use for filtering the Sentinel 2 scenes.
            Defaults to 10.
        input_cache (Path): The directory to use for caching the input data. Defaults to Path("data/cache/input").

        model_files (Path | list[Path]): The path to the models to use for segmentation.
            Can also be a single Path to only use one model. This implies `write_model_outputs=False`
            If a list is provided, will use an ensemble of the models.
        output_data_dir (Path): The "output" directory. Defaults to Path("data/output").
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
            Defaults to Path("data/download/arcticdem").
        tcvis_dir (Path): The directory containing the TCVis data. Defaults to Path("data/download/tcvis").
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
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.
        reflection (int, optional): The reflection padding to use for inference. Defaults to 0.
        binarization_threshold (float, optional): The threshold to binarize the probabilities. Defaults to 0.5.
        mask_erosion_size (int, optional): The size of the disk to use for mask erosion and the edge-cropping.
            Defaults to 10.
        min_object_size (int, optional): The minimum object size to keep in pixel. Defaults to 32.
        quality_level (int | Literal["high_quality", "low_quality", "none"], optional):
            The quality level to use for the segmentation. Can also be an int.
            In this case 0="none" 1="low_quality" 2="high_quality". Defaults to 1.
        export_bands (list[str], optional): The bands to export.
            Can be a list of "probabilities", "binarized", "polygonized", "extent", "thumbnail", "optical", "dem",
            "tcvis" or concrete band-names.
            Defaults to ["probabilities", "binarized", "polygonized", "extent", "thumbnail"].
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.

    """

    aoi_shapefile: Path = None
    start_date: str = None
    end_date: str = None
    max_cloud_cover: int = 10
    max_snow_cover: int | None = 10
    input_cache: Path = Path("data/cache/input")

    def _arcticdem_resolution(self) -> Literal[10]:
        return 10

    @cached_property
    def _s2ids(self) -> list[str]:
        from darts_acquisition import get_gee_s2_sr_scene_ids_from_geodataframe

        return sorted(
            get_gee_s2_sr_scene_ids_from_geodataframe(
                self.aoi_shapefile, self.start_date, self.end_date, self.max_cloud_cover, self.max_snow_cover
            )
        )

    def _get_tile_id(self, tilekey):
        # In case of the GEE tilekey is also the s2id
        return tilekey

    def _tileinfos(self) -> list[tuple[str, Path]]:
        out = []
        for s2id in self._s2ids:
            outpath = self.output_data_dir / s2id
            out.append((s2id, outpath))
        out.sort()
        return out

    def _load_tile(self, tileinfo: RayInputDict) -> "RayDataDict":
        from darts_acquisition import load_gee_s2_sr_scene

        from darts.pipelines._ray_wrapper import RayDataset

        s2id: str = tileinfo["tilekey"]
        output_dir_for_debug_geotiff = None
        if self.debug_data:
            output_dir_for_debug_geotiff = self.output_data_dir / s2id
        tile = load_gee_s2_sr_scene(
            s2id,
            store=self.input_cache,
            offline=self.offline,
            output_dir_for_debug_geotiff=output_dir_for_debug_geotiff,
            device=self.device,
        )
        tile = RayDataset(dataset=tile)
        return {"tile": tile, **tileinfo}

    @staticmethod
    def cli(*, pipeline: "Sentinel2RayPipeline"):
        """Run the sequential pipeline for AOI Sentinel 2 data."""
        pipeline.run()
