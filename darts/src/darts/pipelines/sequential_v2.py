"""Sequential implementation of the v2 pipelines."""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import toml
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

    This class provides the `run` and `prepare_data` methods which are the main entry points for all pipelines.

    This class is meant to be subclassed by the specific pipelines (e.g., PlanetPipeline, Sentinel2Pipeline).
    Subclasses must implement the following abstract methods:
        - `_arcticdem_resolution`: Return the ArcticDEM resolution to use (2, 10, or 32 meters).
        - `_get_tile_id`: Extract a tile identifier from a tilekey.
        - `_tileinfos`: Return a list of (tilekey, output_path) tuples for all tiles to process.
        - `_load_tile`: Load optical data for a given tilekey.
        - `_tile_aoi`: Return a GeoDataFrame representing the area of interest for all tiles.

    Optionally, subclasses can override `_download_tile` to implement data download functionality.

    The subclass must also be a dataclass to fully inherit all parameters of this class.

    Args:
        model_files (list[Path] | None): List of model file paths to use for segmentation.
            If None, will search the default model directory for all .pt files. Defaults to None.
        default_dirs (DefaultPaths): Default directory paths configuration. Defaults to DefaultPaths().
        output_data_dir (Path | None): The output directory for results.
            If None, will use the default output directory based on DARTS paths. Defaults to None.
        arcticdem_dir (Path | None): Directory containing ArcticDEM datacube and extent files.
            If None, will use the default directory based on DARTS paths and resolution. Defaults to None.
        tcvis_dir (Path | None): Directory containing TCVis data.
            If None, will use the default TCVis directory. Defaults to None.
        device (Literal["cuda", "cpu", "auto"] | int | None): Device for computation.
            "cuda" uses GPU 0, int specifies GPU index, "auto" selects free GPU, "cpu" uses CPU.
            Defaults to None (auto-selected).
        ee_project (str | None): Earth Engine project ID. May be omitted if defined in persistent credentials.
            Defaults to None.
        ee_use_highvolume (bool): Whether to use Earth Engine high-volume server. Defaults to True.
        tpi_outer_radius (int): Outer radius in meters for TPI (Topographic Position Index) calculation.
            Defaults to 100.
        tpi_inner_radius (int): Inner radius in meters for TPI calculation. Defaults to 0.
        patch_size (int): Patch size for inference. Defaults to 1024.
        overlap (int): Overlap between patches during inference. Defaults to 256.
        batch_size (int): Batch size for inference. Defaults to 8.
        reflection (int): Reflection padding for inference. Defaults to 0.
        binarization_threshold (float): Threshold for binarizing probabilities. Defaults to 0.5.
        mask_erosion_size (int): Size of disk for mask erosion and inner edge cropping. Defaults to 10.
        edge_erosion_size (int | None): Size for outer edge cropping.
            If None, defaults to `mask_erosion_size`. Defaults to None.
        min_object_size (int): Minimum object size in pixels to keep. Defaults to 32.
        quality_level (int | Literal["high_quality", "low_quality", "none"]): Quality filtering level.
            Can be 0="none", 1="low_quality", 2="high_quality". Defaults to 1.
        export_bands (list[str]): Bands to export, e.g., "probabilities", "binarized", "polygonized",
            "extent", "thumbnail", "optical", "dem", "tcvis", "metadata", or specific band names.
            Defaults to ["probabilities", "binarized", "polygonized", "extent", "thumbnail"].
        write_model_outputs (bool): Whether to save individual model outputs (not just ensemble).
            Automatically set to False if only one model is used. Defaults to False.
        overwrite (bool): Whether to overwrite existing output files. Defaults to False.
        offline (bool): If True, will not attempt to download any missing data. Defaults to False.
        debug_data (bool): If True, writes intermediate data for debugging purposes. Defaults to False.

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
        # The defaults will be overwritten in the respective realizations
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
        """Return the resolution of the ArcticDEM data.

        Returns:
            The ArcticDEM resolution in meters (2, 10, or 32).

        """
        pass

    @abstractmethod
    def _get_tile_id(self, tilekey: Any) -> str:
        """Extract a string identifier from a tilekey.

        Args:
            tilekey: The tilekey (e.g., file path or scene ID).

        Returns:
            A string identifier for the tile.

        """
        pass

    @abstractmethod
    def _tileinfos(self) -> list[tuple[Any, Path]]:
        """Generate list of tiles to process.

        Returns:
            List of tuples containing:
                - tilekey: Anything needed to load the tile (e.g., path or tile ID)
                - output_path: Path to the output directory for this tile

        """
        pass

    @abstractmethod
    def _load_tile(self, tileinfo: Any) -> "xr.Dataset":
        """Load optical data for a given tile.

        Args:
            tileinfo: Information needed to load the tile (from _tileinfos).

        Returns:
            xarray Dataset containing optical bands and masks.

        """
        pass

    @abstractmethod
    def _tile_aoi(self) -> "gpd.GeoDataFrame":
        """Return a GeoDataFrame representing the area of interest for all tiles.

        Returns:
            GeoDataFrame in EPSG:4326 containing geometries for all tiles.

        """
        pass

    def _download_tile(self, tileinfo: Any) -> None:
        """Download optical data for a given tile.

        Optional method that can be overridden by subclasses to implement
        data download functionality for offline processing.

        Args:
            tileinfo: Information needed to download the tile.

        """
        pass

    def _result_metadata(self, tilekey: Any) -> dict:
        """Generate metadata dictionary for export.

        Args:
            tilekey: The tilekey for the current tile.

        Returns:
            Dictionary with DARTS_ prefixed metadata keys and values.

        """
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
        """Validate pipeline configuration.

        Raises:
            ValueError: If no model files or export bands are provided.

        """
        if self.model_files is None or len(self.model_files) == 0:
            raise ValueError("No model files provided. Please provide a list of model files.")
        if len(self.export_bands) == 0:
            raise ValueError("No export bands provided. Please provide a list of export bands.")

    def _dump_config(self) -> str:
        """Save pipeline configuration to TOML file.

        Creates a timestamped configuration file in the output directory.

        Returns:
            Timestamp string used for the configuration filename.

        """
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"Starting pipeline at {current_time}.")

        # Storing the configuration as TOML file
        self.output_data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_data_dir / f"{current_time}.config.toml", "w") as f:
            config = asdict(self)
            # Convert everything to toml serializable
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
        """Create auxiliary data datacubes if they don't exist.

        Args:
            arcticdem: If True, creates ArcticDEM datacube. Defaults to True.
            tcvis: If True, creates TCVis datacube. Defaults to True.

        """
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
        """Load and initialize the ensemble of segmentation models.

        Returns:
            Initialized EnsembleV1 instance with loaded models.

        """
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
        """Check which auxiliary data is required by the ensemble.

        Args:
            ensemble: The loaded ensemble instance.

        Returns:
            Tuple of (needs_arcticdem, needs_tcvis) booleans.

        """
        # Get the ensemble to check which auxiliary data is necessary
        required_bands = ensemble.required_bands
        arcticdem_bands = {"dem", "relative_elevation", "slope", "aspect", "hillshade", "curvature"}
        tcvis_bands = {"tc_brightness", "tc_greenness", "tc_wetness"}
        needs_arcticdem = len(required_bands.intersection(arcticdem_bands)) > 0
        needs_tcvis = len(required_bands.intersection(tcvis_bands)) > 0
        return needs_arcticdem, needs_tcvis

    def prepare_data(self, optical: bool = False, aux: bool = False):
        """Download and prepare data for offline processing.

        Validates configuration, determines data requirements from models,
        and downloads requested data (optical imagery and/or auxiliary data).

        Args:
            optical: If True, downloads optical imagery. Defaults to False.
            aux: If True, downloads auxiliary data (ArcticDEM, TCVis) as needed. Defaults to False.

        Raises:
            KeyboardInterrupt: If user interrupts execution.
            SystemExit: If the process is terminated.
            SystemError: If a system error occurs.

        """
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
                    logger.info("start download ArcticDEM")
                    with timer("Downloading ArcticDEM"):
                        download_arcticdem(aoi, self.arcticdem_dir, resolution=self._arcticdem_resolution())
                if needs_tcvis:
                    logger.info("start download TCVIS")
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
        """Run the complete segmentation pipeline.

        Executes the full pipeline including:
        1. Configuration validation and dumping
        2. Loading ensemble models
        3. Creating/loading auxiliary datacubes
        4. Processing each tile:
           - Loading optical data
           - Loading auxiliary data (ArcticDEM, TCVis) as needed
           - Preprocessing
           - Segmentation
           - Postprocessing
           - Exporting results
        5. Saving results and timing information

        Results are saved to the output directory with timestamped configuration,
        results parquet file, and timing information.

        Raises:
            KeyboardInterrupt: If user interrupts execution.

        """
        self._validate()
        current_time = self._dump_config()

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
                        debug=self.debug_data,
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
    """Pipeline for processing PlanetScope data.

    Processes PlanetScope imagery (both orthotiles and scenes) for RTS segmentation.
    Supports both offline and online processing modes.

    Data Structure:
        Expects PlanetScope data organized as:
        - Orthotiles: `orthotiles_dir/tile_id/scene_id/`
        - Scenes: `scenes_dir/scene_id/`

    Args:
        orthotiles_dir (Path | None): Directory containing PlanetScope orthotiles.
            If None, uses default path from DARTS paths. Defaults to None.
        scenes_dir (Path | None): Directory containing PlanetScope scenes.
            If None, uses default path from DARTS paths. Defaults to None.
        image_ids (list | None): List of image/scene IDs to process.
            If None, processes all images found in orthotiles_dir and scenes_dir. Defaults to None.
        model_files (Path | list[Path] | None): Path(s) to model file(s) for segmentation.
            Single Path implies `write_model_outputs=False`.
            If None, searches default model directory for all .pt files. Defaults to None.
        output_data_dir (Path | None): Output directory for results.
            If None, uses `{default_out}/planet`. Defaults to None.
        arcticdem_dir (Path | None): Directory for ArcticDEM datacube.
            Will be created/downloaded if needed. If None, uses default path. Defaults to None.
        tcvis_dir (Path | None): Directory for TCVis data.
            If None, uses default path. Defaults to None.
        device (Literal["cuda", "cpu", "auto"] | int | None): Computation device.
            "cuda" uses GPU 0, int specifies GPU index, "auto" selects free GPU. Defaults to None.
        ee_project (str | None): Earth Engine project ID.
            May be omitted if defined in persistent credentials. Defaults to None.
        ee_use_highvolume (bool): Whether to use EE high-volume server. Defaults to True.
        tpi_outer_radius (int): Outer radius (m) for TPI calculation. Defaults to 100.
        tpi_inner_radius (int): Inner radius (m) for TPI calculation. Defaults to 0.
        patch_size (int): Patch size for inference. Defaults to 1024.
        overlap (int): Overlap between patches. Defaults to 256.
        batch_size (int): Batch size for inference. Defaults to 8.
        reflection (int): Reflection padding for inference. Defaults to 0.
        binarization_threshold (float): Threshold for binarizing probabilities. Defaults to 0.5.
        mask_erosion_size (int): Disk size for mask erosion and inner edge cropping. Defaults to 10.
        edge_erosion_size (int | None): Size for outer edge cropping.
            If None, uses `mask_erosion_size`. Defaults to None.
        min_object_size (int): Minimum object size (pixels) to keep. Defaults to 32.
        quality_level (int | Literal["high_quality", "low_quality", "none"]): Quality filtering level.
            0="none", 1="low_quality", 2="high_quality". Defaults to 1.
        export_bands (list[str]): Bands to export.
            Can include "probabilities", "binarized", "polygonized", "extent", "thumbnail",
            "optical", "dem", "tcvis", "metadata", or specific band names.
            Defaults to ["probabilities", "binarized", "polygonized", "extent", "thumbnail"].
        write_model_outputs (bool): Save individual model outputs (not just ensemble).
            Defaults to False.
        overwrite (bool): Overwrite existing output files. Defaults to False.
        offline (bool): Skip downloading missing data. Defaults to False.
        debug_data (bool): Write intermediate debugging data. Defaults to False.

    """

    orthotiles_dir: Path | None = None
    scenes_dir: Path | None = None
    image_ids: list = None

    def __post_init__(self):  # noqa: D105
        super().__post_init__()
        self.output_data_dir = self.output_data_dir or (paths.out / "planet")
        self.orthotiles_dir = self.orthotiles_dir or paths.planet_orthotiles()
        self.scenes_dir = self.scenes_dir or paths.planet_scenes()

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
        """Download all necessary data for offline processing.

        Args:
            pipeline: Configured PlanetPipeline instance.
            aux: If True, downloads auxiliary data (ArcticDEM, TCVis). Defaults to False.

        """
        assert not pipeline.offline, "Pipeline must be online to prepare data for offline usage."
        pipeline.__post_init__()
        pipeline.prepare_data(optical=False, aux=aux)

    @staticmethod
    def cli(*, pipeline: "PlanetPipeline"):
        """Run the sequential pipeline for PlanetScope data.

        Args:
            pipeline: Configured PlanetPipeline instance.

        """
        pipeline.__post_init__()
        pipeline.run()


@dataclass
class Sentinel2Pipeline(_BasePipeline):
    """Pipeline for processing Sentinel-2 data.

    Processes Sentinel-2 Surface Reflectance (SR) imagery from either CDSE or Google Earth Engine.
    Supports multiple scene selection methods and flexible filtering options.

    Source Selection:
        The data source is specified via the `raw_data_source` parameter:
        - "cdse": Copernicus Data Space Ecosystem (CDSE)
        - "gee": Google Earth Engine (GEE)

        Both sources require accounts and proper credential setup on the system.

    Scene Selection:
        Scenes can be selected using one of four mutually exclusive methods (priority order):

        1. `scene_ids`: Direct list of Sentinel-2 scene IDs
        2. `scene_id_file`: JSON file containing scene IDs
        3. `tile_ids`: List of Sentinel-2 tile IDs (e.g., "33UVP") with optional filters
        4. `aoi_file`: Shapefile defining area of interest with optional filters

    Filtering Options:
        When using `tile_ids` or `aoi_file`, scenes can be filtered by:
        - Cloud/snow cover: `max_cloud_cover`, `max_snow_cover`
        - Date range: `start_date` and `end_date` (YYYY-MM-DD format)
        - OR specific months/years: `months` (1-12) and `years`

        Note: Date range takes priority over month/year filtering.
        Warning: No temporal filtering may cause rate-limit errors.
        Note: Month/year filtering is experimental and only implemented for CDSE.

    Offline Processing:
        Use `cli_prepare_data` to download data for offline use.
        The `prep_data_scene_id_file` stores scene IDs from queries for offline reuse.

    Args:
        scene_ids (list[str] | None): Direct list of Sentinel-2 scene IDs to process. Defaults to None.
        scene_id_file (Path | None): JSON file containing scene IDs to process. Defaults to None.
        tile_ids (list[str] | None): List of Sentinel-2 tile IDs (requires filtering params). Defaults to None.
        aoi_file (Path | None): Shapefile with area of interest (requires filtering params). Defaults to None.
        start_date (str | None): Start date for filtering (YYYY-MM-DD format). Defaults to None.
        end_date (str | None): End date for filtering (YYYY-MM-DD format). Defaults to None.
        max_cloud_cover (int | None): Maximum cloud cover percentage (0-100). Defaults to 10.
        max_snow_cover (int | None): Maximum snow cover percentage (0-100). Defaults to 10.
        months (list[int] | None): Filter by months (1-12). Defaults to None.
        years (list[int] | None): Filter by years. Defaults to None.
        prep_data_scene_id_file (Path | None): File to store/load scene IDs for offline processing.
            Written during `prepare_data`, read during offline `run`. Defaults to None.
        sentinel2_grid_dir (Path | None): Directory for Sentinel-2 grid shapefiles.
            Used only in `prepare_data` with `tile_ids`. If None, uses default path. Defaults to None.
        raw_data_store (Path | None): Directory for storing raw Sentinel-2 data locally.
            If None, uses default path based on `raw_data_source`. Defaults to None.
        no_raw_data_store (bool): If True, processes data in-memory without local storage.
            Overrides `raw_data_store`. Defaults to False.
        raw_data_source (Literal["gee", "cdse"]): Data source to use. Defaults to "cdse".
        model_files (Path | list[Path] | None): Path(s) to model file(s) for segmentation.
            Single Path implies `write_model_outputs=False`.
            If None, searches default model directory for all .pt files. Defaults to None.
        output_data_dir (Path | None): Output directory for results.
            If None, uses `{default_out}/sentinel2-{raw_data_source}`. Defaults to None.
        arcticdem_dir (Path | None): Directory for ArcticDEM datacube.
            Will be created/downloaded if needed. If None, uses default path. Defaults to None.
        tcvis_dir (Path | None): Directory for TCVis data.
            If None, uses default path. Defaults to None.
        device (Literal["cuda", "cpu", "auto"] | int | None): Computation device.
            "cuda" uses GPU 0, int specifies GPU index, "auto" selects free GPU. Defaults to None.
        ee_project (str | None): Earth Engine project ID.
            May be omitted if defined in persistent credentials. Defaults to None.
        ee_use_highvolume (bool): Whether to use EE high-volume server. Defaults to True.
        tpi_outer_radius (int): Outer radius (m) for TPI calculation. Defaults to 100.
        tpi_inner_radius (int): Inner radius (m) for TPI calculation. Defaults to 0.
        patch_size (int): Patch size for inference. Defaults to 1024.
        overlap (int): Overlap between patches. Defaults to 256.
        batch_size (int): Batch size for inference. Defaults to 8.
        reflection (int): Reflection padding for inference. Defaults to 0.
        binarization_threshold (float): Threshold for binarizing probabilities. Defaults to 0.5.
        mask_erosion_size (int): Disk size for mask erosion and inner edge cropping. Defaults to 10.
        edge_erosion_size (int | None): Size for outer edge cropping.
            If None, uses `mask_erosion_size`. Defaults to None.
        min_object_size (int): Minimum object size (pixels) to keep. Defaults to 32.
        quality_level (int | Literal["high_quality", "low_quality", "none"]): Quality filtering level.
            0="none", 1="low_quality", 2="high_quality". Defaults to 1.
        export_bands (list[str]): Bands to export.
            Can include "probabilities", "binarized", "polygonized", "extent", "thumbnail",
            "optical", "dem", "tcvis", "metadata", or specific band names.
            Defaults to ["probabilities", "binarized", "polygonized", "extent", "thumbnail"].
        write_model_outputs (bool): Save individual model outputs (not just ensemble).
            Defaults to False.
        overwrite (bool): Overwrite existing output files. Defaults to False.
        offline (bool): Skip downloading missing data. Requires pre-downloaded data. Defaults to False.
        debug_data (bool): Write intermediate debugging data to output directory. Defaults to False.

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
    no_raw_data_store: bool = False
    raw_data_source: Literal["gee", "cdse"] = "cdse"

    def __post_init__(self):  # noqa: D105
        logger.debug("Before super")
        super().__post_init__()
        logger.debug("After super")
        self.output_data_dir = self.output_data_dir or (paths.out / f"sentinel2-{self.raw_data_source}")
        self.raw_data_store = self.raw_data_store or paths.sentinel2_raw_data(self.raw_data_source)
        if self.no_raw_data_store:
            self.raw_data_store = None

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
        elif self.tile_ids is not None and self.raw_data_source == "cdse":
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
        elif self.tile_ids is not None and self.raw_data_source == "gee":
            from darts_acquisition import get_gee_s2_sr_scene_ids_from_tile_ids

            logger.debug(f"Getting scene ids from {len(self.tile_ids)} tile ids via GEE.")
            s2ids = get_gee_s2_sr_scene_ids_from_tile_ids(
                self.tile_ids,
                start_date=self.start_date,
                end_date=self.end_date,
                max_cloud_cover=self.max_cloud_cover,
                max_snow_cover=self.max_snow_cover,
            )
        elif self.aoi_file is not None and self.raw_data_source == "cdse":
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
        elif self.aoi_file is not None and self.raw_data_source == "gee":
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

            grid_dir = self.sentinel2_grid_dir or paths.sentinel2_grid()
            grid_file = grid_dir.resolve() / "sentinel_2_index_shapefile.shp"
            if not grid_file.exists():
                download_sentinel_2_grid(grid_dir)
            grid = gpd.read_file(grid_file).to_crs("EPSG:4326")
            return grid[grid["Name"].isin(self.tile_ids)]
        elif self.aoi_file is not None:
            return gpd.read_file(self.aoi_file).to_crs("EPSG:4326")
        else:
            raise ValueError("No valid scene selection method provided.")

        if self.raw_data_source == "cdse":
            from darts_acquisition import get_aoi_from_cdse_scene_ids

            return get_aoi_from_cdse_scene_ids(s2ids)
        else:
            from darts_acquisition import get_aoi_from_gee_scene_ids

            return get_aoi_from_gee_scene_ids(s2ids)

    def _download_tile(self, s2id: str):
        # We default to a path here because the download functions need a path to store the data
        # Note that in the normal load tile function, we can pass None to process in memory
        raw_data_store = self.raw_data_store or paths.sentinel2_raw_data(self.raw_data_source)
        if self.raw_data_source == "gee":
            from darts_acquisition import download_gee_s2_sr_scene

            return download_gee_s2_sr_scene(s2id, store=raw_data_store)
        else:
            from darts_acquisition import download_cdse_s2_sr_scene

            return download_cdse_s2_sr_scene(s2id, store=raw_data_store)

    def _load_tile(self, s2id: str) -> "xr.Dataset":
        output_dir_for_debug_geotiff = None
        if self.debug_data:
            output_dir_for_debug_geotiff = self.output_data_dir / s2id

        if self.raw_data_source == "gee":
            from darts_acquisition import load_gee_s2_sr_scene

            return load_gee_s2_sr_scene(
                s2id,
                store=self.raw_data_store,
                offline=self.offline,
                output_dir_for_debug_geotiff=output_dir_for_debug_geotiff,
                device=self.device,
            )
        else:
            from darts_acquisition import load_cdse_s2_sr_scene

            return load_cdse_s2_sr_scene(
                s2id,
                store=self.raw_data_store,
                offline=self.offline,
                output_dir_for_debug_geotiff=output_dir_for_debug_geotiff,
                device=self.device,
            )

    @staticmethod
    def cli_prepare_data(*, pipeline: "Sentinel2Pipeline", optical: bool = False, aux: bool = False):
        """Download all necessary data for offline processing.

        Queries the data source (CDSE or GEE) for scene IDs and downloads optical and/or auxiliary data.
        Stores scene IDs in `prep_data_scene_id_file` if specified for later offline use.

        Args:
            pipeline: Configured Sentinel2Pipeline instance.
            optical: If True, downloads optical (Sentinel-2) imagery. Defaults to False.
            aux: If True, downloads auxiliary data (ArcticDEM, TCVis). Defaults to False.

        """
        assert not pipeline.offline, "Pipeline must be online to prepare data for offline usage."

        # !: Because of an unknown bug, __post_init__ is not initialized automatically
        pipeline.__post_init__()

        logger.debug(f"Preparing data with {optical=}, {aux=}.")

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
        """Run the sequential pipeline for Sentinel-2 data.

        Args:
            pipeline: Configured Sentinel2Pipeline instance.

        """
        pipeline.__post_init__()
        pipeline.run()


# Cyclopts 4 compatibility -> This complete file would need an architectural rewrite

# planet_cli,
# planet_cli_prepare_data,
# sentinel2_cli,
# sentinel2_cli_prepare_data,


def planet_cli(*, pipeline: PlanetPipeline = PlanetPipeline()):
    """Run the sequential pipeline for PlanetScope data.

    Args:
        pipeline: Configured PlanetPipeline instance.

    """
    PlanetPipeline.cli(pipeline=pipeline)


def planet_cli_prepare_data(*, pipeline: PlanetPipeline = PlanetPipeline(), aux: bool = False):
    """Download all necessary data for offline processing.

    Args:
        pipeline: Configured PlanetPipeline instance.
        aux: If True, downloads auxiliary data (ArcticDEM, TCVis). Defaults to False.

    """
    PlanetPipeline.cli_prepare_data(pipeline=pipeline, aux=aux)


def sentinel2_cli(*, pipeline: Sentinel2Pipeline = Sentinel2Pipeline()):
    """Run the sequential pipeline for Sentinel-2 data.

    Args:
        pipeline: Configured Sentinel2Pipeline instance.

    """
    Sentinel2Pipeline.cli(pipeline=pipeline)


def sentinel2_cli_prepare_data(
    *, pipeline: Sentinel2Pipeline = Sentinel2Pipeline(), optical: bool = False, aux: bool = False
):
    """Download all necessary data for offline processing.

    Args:
        pipeline: Configured Sentinel2Pipeline instance.
        optical: If True, downloads optical (Sentinel-2) imagery. Defaults to False.
        aux: If True, downloads auxiliary data (ArcticDEM, TCVis). Defaults to False.

    """
    Sentinel2Pipeline.cli_prepare_data(pipeline=pipeline, optical=optical, aux=aux)
