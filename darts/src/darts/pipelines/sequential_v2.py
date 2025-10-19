"""Sequential implementation of the v2 pipelines."""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from cyclopts import Parameter
from darts_utils.paths import DefaultPaths, paths

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr
    from darts_ensemble import EnsembleV1

logger = logging.getLogger(__name__)


@Parameter(name="*")
@dataclass
class _BasePipeline(ABC):
    """Base class for all v2 pipelines.

    This class provides the run method which is the main entry point for all pipelines.

    This class is meant to be subclassed by the specific pipelines.
    These pipeliens must implement the _aqdata_generator method.

    The main class must be also a dataclass, to fully inherit all parameter of this class (and the mixins).
    """

    model_files: list[Path] = None
    default_dirs: DefaultPaths = field(default_factory=lambda: DefaultPaths())
    output_data_dir: Path | None = None
    arcticdem_dir: Path | None = None
    tcvis_dir: Path | None = None
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
        self.model_files = self.model_files or list(self.paths.models.glob("*.pt"))
        if self.arcticdem_dir is None:
            arcticdem_resolution = self._arcticdem_resolution()
            self.arcticdem_dir = paths.aux / f"arcticdem{arcticdem_resolution}m.icechunk"
        self.tcvis_dir = self.tcvis_dir or paths.aux / "tcvis.icechunk"
        if self.edge_erosion_size is None:
            self.edge_erosion_size = self.mask_erosion_size

    @property
    @abstractmethod
    def _is_local(self) -> bool:
        pass

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
    def _load_tile(self, tileinfo: Any) -> "xr.Dataset":
        pass

    @abstractmethod
    def _tile_aoi(self) -> "gpd.GeoDataFrame":
        pass

    def _download_tile(self, tileinfo: Any) -> None:
        pass

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

    def _validate(self):
        if self.model_files is None or len(self.model_files) == 0:
            raise ValueError("No model files provided. Please provide a list of model files.")
        if len(self.export_bands) == 0:
            raise ValueError("No export bands provided. Please provide a list of export bands.")

    def _dump_config(self) -> str:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"Starting pipeline at {current_time}.")

        # Storing the configuration as JSON file
        self.output_data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_data_dir / f"{current_time}.config.json", "w") as f:
            config = asdict(self)
            # Convert everything to json serializable
            for key, value in config.items():
                if isinstance(value, Path):
                    config[key] = str(value.resolve())
                elif isinstance(value, list):
                    config[key] = [str(v.resolve()) if isinstance(v, Path) else v for v in value]
            json.dump(config, f)
        return current_time

    def _create_auxiliary_datacubes(self, arcticdem: bool = True, tcvis: bool = True):
        import smart_geocubes

        from darts.utils.logging import LoggingManager

        # Create the datacubes if they do not exist
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

    def _load_ensemble(self) -> "EnsembleV1":
        import torch
        from darts_ensemble import EnsembleV1

        # determine models to use
        if isinstance(self.model_files, Path):
            self.model_files = [self.model_files]
        if len(self.model_files) == 1:
            self.write_model_outputs = False
        models = {model_file.stem: model_file for model_file in self.model_files}
        ensemble = EnsembleV1(models, device=torch.device(self.device))
        return ensemble

    def _check_aux_needs(self, ensemble: "EnsembleV1") -> tuple[bool, bool]:
        # Get the ensemble to check which auxiliary data is necessary
        required_bands = ensemble.required_bands
        arcticdem_bands = {"dem", "relative_elevation", "slope", "aspect", "hillshade", "curvature"}
        tcvis_bands = {"tc_brightness", "tc_greenness", "tc_wetness"}
        needs_arcticdem = len(required_bands.intersection(arcticdem_bands)) > 0
        needs_tcvis = len(required_bands.intersection(tcvis_bands)) > 0
        return needs_arcticdem, needs_tcvis

    def prepare_data(self, optical: bool = False, aux: bool = False):
        assert optical or aux, "Nothing to prepare. Please set optical and/or aux to True."

        self._validate()
        self._dump_config()

        from darts.utils.cuda import debug_info

        debug_info()

        from darts_acquisition import download_arcticdem, download_tcvis
        from stopuhr import Chronometer

        from darts.utils.cuda import decide_device
        from darts.utils.earthengine import init_ee

        timer = Chronometer(printer=logger.debug)
        self.device = decide_device(self.device)

        if aux:
            # Get the ensemble to check which auxiliary data is necessary
            ensemble = self._load_ensemble()
            needs_arcticdem, needs_tcvis = self._check_aux_needs(ensemble)

            if not needs_arcticdem and not needs_tcvis:
                logger.warning("No auxiliary data required by the models. Skipping download of auxiliary data...")
            else:
                logger.info(f"Models {needs_tcvis=} {needs_arcticdem=}.")
                self._create_auxiliary_datacubes(arcticdem=needs_arcticdem, tcvis=needs_tcvis)

                # Predownload auxiliary
                aoi = self._tile_aoi()
                if needs_arcticdem:
                    with timer("Downloading ArcticDEM"):
                        download_arcticdem(aoi, self.arcticdem_dir, resolution=self._arcticdem_resolution())
                if needs_tcvis:
                    init_ee(self.ee_project, self.ee_use_highvolume)
                    with timer("Downloading TCVis"):
                        download_tcvis(aoi, self.tcvis_dir)

        # Predownload tiles if optical flag is set
        if not optical:
            return

        # Iterate over all the data
        with timer("Loading Optical"):
            tileinfo = self._tileinfos()
            n_tiles = 0
            logger.info(f"Found {len(tileinfo)} tiles to download.")
            for i, (tilekey, _) in enumerate(tileinfo):
                tile_id = self._get_tile_id(tilekey)
                try:
                    self._download_tile(tilekey)
                    n_tiles += 1
                    logger.info(f"Downloaded sample {i + 1} of {len(tileinfo)} '{tilekey}' ({tile_id=}).")
                except (KeyboardInterrupt, SystemError, SystemExit) as e:
                    logger.warning(f"{type(e).__name__} detected.\nExiting...")
                    raise e
                except Exception as e:
                    logger.warning(f"Could not process '{tilekey}' ({tile_id=}).\nSkipping...")
                    logger.exception(e)
            else:
                logger.info(f"Downloaded {n_tiles} tiles.")

    def run(self):  # noqa: C901
        current_time = self._validate()
        self._dump_config()

        from darts.utils.cuda import debug_info

        debug_info()

        import pandas as pd
        from darts_acquisition import load_arcticdem, load_tcvis
        from darts_export import export_tile, missing_outputs
        from darts_postprocessing import prepare_export
        from darts_preprocessing import preprocess_v2
        from stopuhr import Chronometer, stopwatch

        from darts.utils.cuda import decide_device
        from darts.utils.earthengine import init_ee

        timer = Chronometer(printer=logger.debug)
        self.device = decide_device(self.device)

        if not self.offline:
            init_ee(self.ee_project, self.ee_use_highvolume)

        self._create_auxiliary_datacubes()

        # determine models to use
        ensemble = self._load_ensemble()
        ensemble_subsets = ensemble.model_names
        needs_arcticdem, needs_tcvis = self._check_aux_needs(ensemble)

        # Iterate over all the data
        tileinfo = self._tileinfos()
        n_tiles = 0
        logger.info(f"Found {len(tileinfo)} tiles to process.")
        results = []
        for i, (tilekey, outpath) in enumerate(tileinfo):
            tile_id = self._get_tile_id(tilekey)
            try:
                if not self.overwrite:
                    mo = missing_outputs(outpath, bands=self.export_bands, ensemble_subsets=ensemble_subsets)
                    if mo == "none":
                        logger.info(f"Tile {tile_id} already processed. Skipping...")
                        continue
                    if mo == "some":
                        logger.warning(
                            f"Tile {tile_id} seems to be already processed, "
                            "but some of the requested outputs are missing. "
                            "Skipping because overwrite=False..."
                        )
                        continue

                with timer("Loading Optical", log=False):
                    tile = self._load_tile(tilekey)

                if needs_arcticdem:
                    with timer("Loading ArcticDEM", log=False):
                        arcticdem_resolution = self._arcticdem_resolution()
                        arcticdem = load_arcticdem(
                            tile.odc.geobox,
                            self.arcticdem_dir,
                            resolution=arcticdem_resolution,
                            buffer=ceil(self.tpi_outer_radius / arcticdem_resolution * sqrt(2)),
                            offline=self.offline,
                        )
                else:
                    arcticdem = None

                if needs_tcvis:
                    with timer("Loading TCVis", log=False):
                        tcvis = load_tcvis(tile.odc.geobox, self.tcvis_dir, offline=self.offline)
                else:
                    tcvis = None

                with timer("Preprocessing", log=False):
                    tile = preprocess_v2(
                        tile,
                        arcticdem,
                        tcvis,
                        self.tpi_outer_radius,
                        self.tpi_inner_radius,
                        self.device,
                    )

                with timer("Segmenting", log=False):
                    tile = ensemble.segment_tile(
                        tile,
                        patch_size=self.patch_size,
                        overlap=self.overlap,
                        batch_size=self.batch_size,
                        reflection=self.reflection,
                        keep_inputs=self.write_model_outputs,
                    )

                with timer("Postprocessing", log=False):
                    tile = prepare_export(
                        tile,
                        bin_threshold=self.binarization_threshold,
                        mask_erosion_size=self.mask_erosion_size,
                        min_object_size=self.min_object_size,
                        quality_level=self.quality_level,
                        ensemble_subsets=ensemble_subsets if self.write_model_outputs else [],
                        device=self.device,
                        edge_erosion_size=self.edge_erosion_size,
                    )

                export_metadata = self._result_metadata(tilekey)

                with timer("Exporting", log=False):
                    export_tile(
                        tile,
                        outpath,
                        bands=self.export_bands,
                        ensemble_subsets=ensemble_subsets if self.write_model_outputs else [],
                        metadata=export_metadata,
                    )

                n_tiles += 1
                results.append(
                    {
                        "tile_id": tile_id,
                        "output_path": str(outpath.resolve()),
                        "status": "success",
                        "error": None,
                    }
                )
                logger.info(f"Processed sample {i + 1} of {len(tileinfo)} '{tilekey}' ({tile_id=}).")
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected.\nExiting...")
                raise KeyboardInterrupt
            except Exception as e:
                logger.warning(f"Could not process '{tilekey}' ({tile_id=}).\nSkipping...")
                logger.exception(e)
                results.append(
                    {
                        "tile_id": tile_id,
                        "output_path": str(outpath.resolve()),
                        "status": "failed",
                        "error": str(e),
                    }
                )
            finally:
                if len(results) > 0:
                    pd.DataFrame(results).to_parquet(self.output_data_dir / f"{current_time}.results.parquet")
                if len(timer.durations) > 0:
                    timer.export().to_parquet(self.output_data_dir / f"{current_time}.timer.parquet")
                if len(stopwatch.durations) > 0:
                    stopwatch.export().to_parquet(self.output_data_dir / f"{current_time}.stopwatch.parquet")
        else:
            logger.info(f"Processed {n_tiles} tiles to {self.output_data_dir.resolve()}.")
            timer.summary(printer=logger.info)


# =============================================================================
# Source Pipelines
# =============================================================================
@dataclass
class PlanetPipeline(_BasePipeline):
    """Pipeline for PlanetScope data.

    Args:
        orthotiles_dir (Path): The directory containing the PlanetScope orthotiles.
        scenes_dir (Path): The directory containing the PlanetScope scenes.
        image_ids (list): The list of image ids to process. If None, all images in the directory will be processed.


        model_files (Path | list[Path] | None, optional): The path to the models to use for segmentation.
            Can also be a single Path to only use one model. This implies `write_model_outputs=False`
            If a list is provided, will use an ensemble of the models.
            If None, will search the default model directory based on the DARTS paths for all .pt files.
            Defaults to None.
        output_data_dir (Path | None, optional): The "output" directory.
            If None, will use the default output directory based on the DARTS paths.
            Defaults to None.
        arcticdem_dir (Path | None, optional): The directory containing the ArcticDEM data
            (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
            If None, will use the default auxiliary directory based on the DARTS paths.
            Defaults to None.
        tcvis_dir (Path | None, optional): The directory containing the TCVis data.
            If None, will use the default TCVis directory based on the DARTS paths.
            Defaults to None.
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
        edge_erosion_size (int, optional): If the edge-cropping should have a different witdth, than the (inner) mask
            erosion, set it here. Defaults to `mask_erosion_size`.
        min_object_size (int, optional): The minimum object size to keep in pixel. Defaults to 32.
        quality_level (int | Literal["high_quality", "low_quality", "none"], optional):
            The quality level to use for the segmentation. Can also be an int.
            In this case 0="none" 1="low_quality" 2="high_quality". Defaults to 1.
        export_bands (list[str], optional): The bands to export.
            Can be a list of "probabilities", "binarized", "polygonized", "extent", "thumbnail", "optical", "dem",
            "tcvis", "metadata" or concrete band-names.
            Defaults to ["probabilities", "binarized", "polygonized", "extent", "thumbnail"].
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        offline (bool, optional): If True, will not attempt to download any missing data. Defaults to False.
        debug_data (bool, optional): If True, will write intermediate data for debugging purposes to output.
            Defaults to False.

    """

    orthotiles_dir: Path | None = None
    scenes_dir: Path | None = None
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
        self.orthotiles_dir = self.orthotiles_dir or paths.input / "planet" / "PSOrthoTile"
        self.scenes_dir = self.scenes_dir or paths.input / "planet" / "PSScene"
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

    def _tile_aoi(self) -> "gpd.GeoDataFrame":
        import geopandas as gpd
        from darts_acquisition import get_planet_geometry

        tileinfos = self._tileinfos()
        aoi = []
        for fpath, _ in tileinfos:
            geom = get_planet_geometry(fpath)
            aoi.append({"tilekey": fpath, "geometry": geom.to_crs("EPSG:4326").geom})
        aoi = gpd.GeoDataFrame(aoi, geometry="geometry", crs="EPSG:4326")
        return aoi

    def _load_tile(self, fpath: Path) -> "xr.Dataset":
        import xarray as xr
        from darts_acquisition import load_planet_masks, load_planet_scene

        optical = load_planet_scene(fpath)
        data_masks = load_planet_masks(fpath)
        tile = xr.merge([optical, data_masks])
        return tile

    @staticmethod
    def cli_prepare_data(*, pipeline: "PlanetPipeline", aux: bool = False):
        """Download all necessary data for offline processing."""
        assert not pipeline.offline, "Pipeline must be online to prepare data for offline usage."
        pipeline.prepare_data(optical=False, aux=aux)

    @staticmethod
    def cli(*, pipeline: "PlanetPipeline"):
        """Run the sequential pipeline for Planet data."""
        pipeline.run()


@dataclass
class Sentinel2Pipeline(_BasePipeline):
    """Pipeline for Sentinel 2.

    ### Source selection

        Because of historical reasons, both the scene source can be specified via the `raw_data_source` parameter.
        Currently, two sources are supported:

            - "cdse": The Copernicus Data and Exploitation Service (CDSE).
            - "gee": Google Earth Engine (GEE).

        Please note that both sources need accounts and the credentials need to be setup on the system respectively.

    ### Scene selection

        There are 4 ways to select the scenes to process:

        1. Provide a list of scene ids via the `scene_ids` parameter.
        2. Provide a file containing a list of scene ids via the `scene_id_file` parameter.
        3. Provide a list of tile ids via the `tile_ids` parameter along with filtering parameters.
        4. Provide an area of interest shapefile via the `aoi_file` parameter along with filtering parameters.

        The selection methods are mutually exclusive and will be used in the order listed above.

    ### Filtering and Date selection

        One can filter the scenes based on the cloud cover and snow cover percentage
        using the `max_cloud_cover` and `max_snow_cover` parameters.

        A temporal filtering can either be applied by passing a start and end date
        via the `start_date` and `end_date` parameters,
        or by providing a list of months and/or years via the `months` and `years` parameters.
        Again, these two selection methods are mutually exclusive and the date range will be used first if provided.
        If no temporal filtering is applied, all available scenes will be selected,
        which may cause rate-limit errors from CDSE or GEE.
        Also note, that the year+month filtering is not well tested and only implemented for CDSE at the moment.

    Args:
        scene_ids (list[str] | None): A list of Sentinel 2 scene ids to process.
        scene_id_file (Path | None): A file containing a list of Sentinel 2 scene ids to process.
        tile_ids (list[str] | None): A list of Sentinel 2 tile ids to process.
        aoi_file (Path | None): The shapefile containing the area of interest.
        start_date (str): The start date of the time series in YYYY-MM-DD format.
        end_date (str): The end date of the time series in YYYY-MM-DD format.
        max_cloud_cover (int): The maximum cloud cover percentage to use for filtering the Sentinel 2 scenes.
        max_snow_cover (int): The maximum snow cover percentage to use for filtering the Sentinel 2 scenes.
        months (list[int] | None): A list of months (1-12) to use for filtering the Sentinel 2 scenes.
        years (list[int] | None): A list of years to use for filtering the Sentinel 2 scenes.
            Defaults to 10.
        prep_data_scene_id_file (Path | None): A file containing a list of Sentinel 2 scene ids to process.
            This is only used for offline processing to avoid querying the data source again.
            If None, will not use any pre-processed scene ids.
            Defaults to None.
        sentinel2_grid_dir (Path | None): The directory to use for storing the Sentinel 2 grid files.
            This is only used for the prep-data step and not in normal mode.
            If None, will use the default auxiliary directory based on the DARTS paths.
            Defaults to None.
        raw_data_store (Path | None): The directory to use for storing the raw Sentinel 2 data locally.
            If None, will not store any data locally and process only in memory.
            Defaults to None.
        raw_data_source (Literal["gee", "cdse"]): The source to use for downloading the Sentinel 2 data.
        model_files (Path | list[Path] | None, optional): The path to the models to use for segmentation.
            Can also be a single Path to only use one model. This implies `write_model_outputs=False`
            If a list is provided, will use an ensemble of the models.
            If None, will search the default model directory based on the DARTS paths for all .pt files.
            Defaults to None.
        output_data_dir (Path | None, optional): The "output" directory.
            If None, will use the default output directory based on the DARTS paths.
            Defaults to None.
        arcticdem_dir (Path | None, optional): The directory containing the ArcticDEM data
            (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
            If None, will use the default auxiliary directory based on the DARTS paths.
            Defaults to None.
        tcvis_dir (Path | None, optional): The directory containing the TCVis data.
            If None, will use the default TCVis directory based on the DARTS paths.
            Defaults to None.
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
        edge_erosion_size (int, optional): If the edge-cropping should have a different witdth, than the (inner) mask
            erosion, set it here. Defaults to `mask_erosion_size`.
        min_object_size (int, optional): The minimum object size to keep in pixel. Defaults to 32.
        quality_level (int | Literal["high_quality", "low_quality", "none"], optional):
            The quality level to use for the segmentation. Can also be an int.
            In this case 0="none" 1="low_quality" 2="high_quality". Defaults to 1.
        export_bands (list[str], optional): The bands to export.
            Can be a list of "probabilities", "binarized", "polygonized", "extent", "thumbnail", "optical", "dem",
            "tcvis", "metadata" or concrete band-names.
            Defaults to ["probabilities", "binarized", "polygonized", "extent", "thumbnail"].
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        offline (bool, optional): If True, will not attempt to download any missing data. Defaults to False.
        debug_data (bool, optional): If True, will write intermediate data for debugging purposes to output.
            Defaults to False.

    """

    # Scene selection
    scene_ids: list[str] | None = None
    scene_id_file: Path | None = None
    tile_ids: list[str] | None = None
    aoi_file: Path | None = None
    # Scene selection filters (only used with tile_ids and aoi_file)
    start_date: str | None = None
    end_date: str | None = None
    max_cloud_cover: int | None = 10
    max_snow_cover: int | None = 10
    months: list[int] | None = None
    years: list[int] | None = None
    # For offline use
    prep_data_scene_id_file: Path | None = None
    sentinel2_grid_dir: Path | None = None
    raw_data_store: Path | None = None
    raw_data_source: Literal["gee", "cdse"] = "cdse"

    def _arcticdem_resolution(self) -> Literal[10]:
        return 10

    def _warn_invalid_selectors(self):
        selectors = ["scene_ids", "scene_id_file", "tile_ids", "aoi_file"]
        user_selectors = [s for s in selectors if getattr(self, s) is not None]
        if len(user_selectors) > 1:
            logger.warning(
                f"Multiple scene selection methods provided: {user_selectors}. "
                "Using only the first one in the order of scene_ids, scene_id_file, tile_ids, aoi_file."
            )

    def _get_s2ids(self) -> list[str]:
        # Logic:
        # Offline: Check for prep_data_scene_id_file first, then scene_ids, then scene_id_file,
        # raise error if tile_ids or aoi_file used
        # Online: Check for scene_ids first, then scene_id_file, then tile_ids, then aoi_file

        if self.offline and self.prep_data_scene_id_file is not None and self.prep_data_scene_id_file.exists():
            logger.debug(f"Using scene id file at {self.prep_data_scene_id_file=} for offline processing.")
            s2ids: list[str] = json.loads(self.prep_data_scene_id_file.read_text())
            return s2ids

        self._warn_invalid_selectors()
        if self.scene_ids is not None:
            logger.debug(f"Using {len(self.scene_ids)} provided scene ids for processing.")
            return self.scene_ids
        elif self.scene_id_file is not None:
            logger.debug(f"Loading scene ids from file {self.scene_id_file=}.")
            s2ids: list[str] = json.loads(self.scene_id_file.read_text())
            return s2ids
        elif self.tile_ids is not None and self.s2_source == "cdse":
            from darts_acquisition import get_cdse_s2_sr_scene_ids_from_tile_ids

            logger.debug(f"Getting scene ids from {len(self.tile_ids)} tile ids via CDSE.")
            s2ids = get_cdse_s2_sr_scene_ids_from_tile_ids(
                self.tile_ids,
                start_date=self.start_date,
                end_date=self.end_date,
                max_cloud_cover=self.max_cloud_cover,
                max_snow_cover=self.max_snow_cover,
                months=self.months,
                years=self.years,
            ).keys()
        elif self.tile_ids is not None and self.s2_source == "gee":
            from darts_acquisition import get_gee_s2_sr_scene_ids_from_tile_ids

            logger.debug(f"Getting scene ids from {len(self.tile_ids)} tile ids via GEE.")
            s2ids = get_gee_s2_sr_scene_ids_from_tile_ids(
                self.tile_ids,
                start_date=self.start_date,
                end_date=self.end_date,
                max_cloud_cover=self.max_cloud_cover,
                max_snow_cover=self.max_snow_cover,
            )
        elif self.aoi_file is not None and self.s2_source == "cdse":
            from darts_acquisition import get_cdse_s2_sr_scene_ids_from_geodataframe

            logger.debug(f"Getting scene ids from AOI file {self.aoi_file=} via CDSE.")
            s2ids = get_cdse_s2_sr_scene_ids_from_geodataframe(
                self.aoi_file,
                start_date=self.start_date,
                end_date=self.end_date,
                max_cloud_cover=self.max_cloud_cover,
                max_snow_cover=self.max_snow_cover,
                months=self.months,
                years=self.years,
            ).keys()
        elif self.aoi_file is not None and self.s2_source == "gee":
            from darts_acquisition import get_gee_s2_sr_scene_ids_from_geodataframe

            logger.debug(f"Getting scene ids from AOI file {self.aoi_file=} via GEE.")
            s2ids = get_gee_s2_sr_scene_ids_from_geodataframe(
                self.aoi_file,
                start_date=self.start_date,
                end_date=self.end_date,
                max_cloud_cover=self.max_cloud_cover,
                max_snow_cover=self.max_snow_cover,
            )
        else:
            logger.error("No valid scene selection method provided.")
            raise ValueError("No valid scene selection method provided.")

        s2ids = sorted(set(s2ids))

        # Note: This only happens if tile_ids or aoi_file were used
        if self.prep_data_scene_id_file is not None:
            logger.debug(f"Storing scene ids to file {self.prep_data_scene_id_file=} for offline processing.")
            self.prep_data_scene_id_file.write_text(json.dumps(s2ids))

        return s2ids

    def _get_tile_id(self, tilekey):
        # In case of the GEE tilekey is also the s2id
        return tilekey

    def _tileinfos(self) -> list[tuple[str, Path]]:
        out = []
        for s2id in self._get_s2ids():
            outpath = self.output_data_dir / s2id
            out.append((s2id, outpath))
        out.sort()
        return out

    def _tile_aoi(self) -> "gpd.GeoDataFrame":
        import geopandas as gpd

        assert not self.offline, "AOI extraction not possible in offline mode without aoi_file."

        if self.scene_ids is not None:
            s2ids = self.scene_ids
        elif self.scene_id_file is not None:
            s2ids = json.loads(self.scene_id_file.read_text())
        elif self.tile_ids is not None:
            from darts_acquisition import download_sentinel_2_grid

            grid_dir = self.sentinel2_grid_dir or paths.aux / "sentinel2_grid"
            grid_file = grid_dir.resolve() / "sentinel_2_index_shapefile.shp"
            if not grid_file.exists():
                download_sentinel_2_grid(grid_dir)
            grid = gpd.read_file(grid_file).to_crs("EPSG:4326")
            return grid[grid["Name"].isin(self.tile_ids)]
        elif self.aoi_file is not None:
            return gpd.read_file(self.aoi_file).to_crs("EPSG:4326")
        else:
            raise ValueError("No valid scene selection method provided.")

        if self.s2_source == "cdse":
            from darts_acquisition import get_aoi_from_cdse_scene_ids

            return get_aoi_from_cdse_scene_ids(s2ids)
        else:
            from darts_acquisition import get_aoi_from_gee_scene_ids

            return get_aoi_from_gee_scene_ids(s2ids)

    def _download_tile(self, s2id: str):
        self.s2_download_cache = self.s2_download_cache or paths.input / self.s2_source
        if self.s2_source == "gee":
            from darts_acquisition import download_gee_s2_sr_scene

            return download_gee_s2_sr_scene(s2id, store=self.s2_download_cache)
        else:
            from darts_acquisition import download_cdse_s2_sr_scene

            return download_cdse_s2_sr_scene(s2id, store=self.s2_download_cache)

    def _load_tile(self, s2id: str) -> "xr.Dataset":
        self.s2_download_cache = self.s2_download_cache or paths.input / self.s2_source
        if self.s2_source == "gee":
            from darts_acquisition import load_gee_s2_sr_scene

            return load_gee_s2_sr_scene(s2id, store=self.s2_download_cache, offline=self.offline)
        else:
            from darts_acquisition import load_cdse_s2_sr_scene

            return load_cdse_s2_sr_scene(s2id, store=self.s2_download_cache, offline=self.offline)

    @staticmethod
    def cli_prepare_data(*, pipeline: "Sentinel2Pipeline", optical: bool = False, aux: bool = False):
        """Download all necessary data for offline processing."""
        assert not pipeline.offline, "Pipeline must be online to prepare data for offline usage."

        if pipeline.prep_data_scene_id_file is not None:
            if pipeline.prep_data_scene_id_file.exists():
                logger.warning(
                    f"Prep-data scene id file {pipeline.prep_data_scene_id_file=} already exists. "
                    "It will be overwritten."
                )
                pipeline.prep_data_scene_id_file.unlink()
        pipeline.prepare_data(optical=optical, aux=aux)

    @staticmethod
    def cli(*, pipeline: "Sentinel2Pipeline"):
        """Run the sequential pipeline for Sentinel 2 data."""
        pipeline.run()
