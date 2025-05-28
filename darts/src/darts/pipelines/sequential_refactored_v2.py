"""Sequential implementation of the v2 pipelines."""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import cached_property
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from cyclopts import Parameter

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


@Parameter(name="*")
@dataclass
class _BasePipelineRefactored(ABC):
    """Base class for all v2 pipelines.

    This class provides the run method which is the main entry point for all pipelines.

    This class is meant to be subclassed by the specific pipelines.
    These pipeliens must implement the _aqdata_generator method.

    The main class must be also a dataclass, to fully inherit all parameter of this class (and the mixins).
    """

    model_files: list[Path] = None
    output_data_dir: Path = Path("data/output")
    arcticdem_dir: Path = Path("data/download/arcticdem")
    tcvis_dir: Path = Path("data/download/tcvis")
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
    min_object_size: int = 32
    quality_level: int | Literal["high_quality", "low_quality", "none"] = 1
    export_bands: list[str] = field(
        default_factory=lambda: ["probabilities", "binarized", "polygonized", "extent", "thumbnail"]
    )
    write_model_outputs: bool = False
    overwrite: bool = False

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

    def _process_tile(self, i, tilekey, outpath, tileinfo, models, timer, ensemble, n_tiles, current_time, results):
        """Process a single tile in the pipeline.

        Args:
            tilekey: The tile identifier (could be path, ID, etc.)
            outpath: Path where output should be saved
            models: Dictionary of model names and paths
            timer: Chronometer instance for timing operations

        Returns:
            Dictionary containing processing results for this tile
        """
        from darts_acquisition import load_arcticdem, load_tcvis
        from darts_export import export_tile, missing_outputs
        from darts_postprocessing import prepare_export
        from darts_preprocessing import preprocess_legacy_fast
        import pandas as pd
        # TODO create timer here

        tile_id = self._get_tile_id(tilekey)
        result = {
            "tile_id": tile_id,
            "output_path": str(outpath.resolve()),
            "status": "failed",  # Default to failed, will update if successful
            "error": None,
        }
        print(f"First result in process tile method")
        print(result)
        try:
            if not self.overwrite:
                mo = missing_outputs(outpath, bands=self.export_bands, ensemble_subsets=models.keys())
                if mo == "none":
                    logger.info(f"Tile {tile_id} already processed. Skipping...")
                    result["status"] = "skipped"
                    return result
                if mo == "some":
                    logger.warning(
                        f"Tile {tile_id} already processed. Some outputs are missing."
                        " Skipping because overwrite=False..."
                    )
                    result["status"] = "skipped_partial"
                    return result

            with timer("Loading optical data", log=False):
                tile = self._load_tile(tilekey)
                print("Loaded optical data")
                print(tile)
            with timer("Loading ArcticDEM", log=False):
                arcticdem = load_arcticdem(
                    tile.odc.geobox,
                    self.arcticdem_dir,
                    resolution=self._arcticdem_resolution(),
                    buffer=ceil(self.tpi_outer_radius / 2 * sqrt(2)),
                )
                print(f"Loaded arctic dem")
                print(arcticdem)
            with timer("Loading TCVis", log=False):
                tcvis = load_tcvis(tile.odc.geobox, self.tcvis_dir)
                print("Loaded tcvis")
                print(tcvis)
            with timer("Preprocessing tile", log=False):
                tile = preprocess_legacy_fast(
                    tile,
                    arcticdem,
                    tcvis,
                    self.tpi_outer_radius,
                    self.tpi_inner_radius,
                    self.device,
                )
                print(f"Preprocessed tile")
                print(tile)
            with timer("Segmenting", log=False):
                tile = ensemble.segment_tile(
                    tile,
                    patch_size=self.patch_size,
                    overlap=self.overlap,
                    batch_size=self.batch_size,
                    reflection=self.reflection,
                    keep_inputs=self.write_model_outputs,
                )
                print("Segmented tile")
                print(tile)
            with timer("Postprosessing", log=False):
                tile = prepare_export(
                    tile,
                    bin_threshold=self.binarization_threshold,
                    mask_erosion_size=self.mask_erosion_size,
                    min_object_size=self.min_object_size,
                    quality_level=self.quality_level,
                    ensemble_subsets=models.keys() if self.write_model_outputs else [],
                    device=self.device,
                )
                print("Postprocessing")
                print(tile)
            with timer("Exporting", log=False):
                print("exporting to outpath", outpath)
                export_tile(
                    tile,
                    outpath,
                    bands=self.export_bands,
                    ensemble_subsets=models.keys() if self.write_model_outputs else [],
                )
                print("exporting")
                print(tile)
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
            raise
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
            print("in finally part")
            print(len(results))
            print('the length of the results')
            if len(results) > 0:
                pd.DataFrame(results).to_parquet(self.output_data_dir / f"{current_time}.results.parquet")
            if len(timer.durations) > 0:
                timer.export().to_parquet(self.output_data_dir / f"{current_time}.stopuhr.parquet")
            return (results, n_tiles)

    def run(self):  # noqa: C901
        if self.model_files is None or len(self.model_files) == 0:
            raise ValueError("No model files provided. Please provide a list of model files.")
        if len(self.export_bands) == 0:
            raise ValueError("No export bands provided. Please provide a list of export bands.")

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"Starting pipeline at {current_time}.")

        # TODO commented out
        # Storing the configuration as JSON file
        # self.output_data_dir.mkdir(parents=True, exist_ok=True)
        # with open(self.output_data_dir / f"{current_time}.config.json", "w") as f:
        #     config = asdict(self)
        #     # Convert everything to json serializable
        #     for key, value in config.items():
        #         if isinstance(value, Path):
        #             config[key] = str(value.resolve())
        #         elif isinstance(value, list):
        #             config[key] = [str(v.resolve()) if isinstance(v, Path) else v for v in value]
        #     json.dump(config, f)

        from stopuhr import Chronometer

        timer = Chronometer(printer=logger.debug)

        from darts.utils.cuda import debug_info

        debug_info()

        from darts.utils.earthengine import init_ee

        init_ee(self.ee_project, self.ee_use_highvolume)

        import pandas as pd
        import smart_geocubes
        import torch
        from darts_acquisition import load_arcticdem, load_tcvis
        from darts_ensemble import EnsembleV1
        from darts_export import export_tile, missing_outputs
        from darts_postprocessing import prepare_export
        from darts_preprocessing import preprocess_legacy_fast

        from darts.utils.cuda import decide_device
        from darts.utils.logging import LoggingManager

        self.device = decide_device(self.device)

        # determine models to use
        if isinstance(self.model_files, Path):
            self.model_files = [self.model_files]
            self.write_model_outputs = False
        models = {model_file.stem: model_file for model_file in self.model_files}
        ensemble = EnsembleV1(models, device=torch.device(self.device))

        # Create the datacubes if they do not exist
        LoggingManager.apply_logging_handlers("smart_geocubes")
        arcticdem_resolution = self._arcticdem_resolution()
        if arcticdem_resolution == 2:
            accessor = smart_geocubes.ArcticDEM2m(self.arcticdem_dir)
        elif arcticdem_resolution == 10:
            accessor = smart_geocubes.ArcticDEM10m(self.arcticdem_dir)
        if not accessor.created:
            accessor.create(overwrite=False)
        accessor = smart_geocubes.TCTrend(self.tcvis_dir)
        if not accessor.created:
            accessor.create(overwrite=False)

        # Iterate over all the data
        tileinfo = self._tileinfos()
        all_tileinfo = enumerate(tileinfo)
        print("tile info and enumerate tile info")
        print(tileinfo)
        print(type(tileinfo))
        print(all_tileinfo)
        print(type(all_tileinfo))
        print('the tile key')
        n_tiles = 0
        logger.info(f"Found {len(tileinfo)} tiles to process.")
        #     def _process_tile(self, i, tilekey, outpath, tileinfo, models, timer, n_tiles, current_time, results):



        results = []

        for i, (tilekey, outpath) in enumerate(tileinfo):
            print(f"Using new process tile method")
            (results, n_tiles) = self._process_tile( i, tilekey, outpath, tileinfo, models, timer, ensemble, n_tiles,current_time, results)
            print('got new result')
            print(results)
            print(n_tiles)

        # for i, (tilekey, outpath) in enumerate(tileinfo):
        #     tile_id = self._get_tile_id(tilekey)
        #     try:
        #         if not self.overwrite:
        #             mo = missing_outputs(outpath, bands=self.export_bands, ensemble_subsets=models.keys())
        #             if mo == "none":
        #                 logger.info(f"Tile {tile_id} already processed. Skipping...")
        #                 continue
        #             if mo == "some":
        #                 logger.warning(
        #                     f"Tile {tile_id} already processed. Some outputs are missing."
        #                     " Skipping because overwrite=False..."
        #                 )
        #                 continue
        #
        #         with timer("Loading optical data", log=False):
        #             tile = self._load_tile(tilekey)
        #         with timer("Loading ArcticDEM", log=False):
        #             arcticdem = load_arcticdem(
        #                 tile.odc.geobox,
        #                 self.arcticdem_dir,
        #                 resolution=arcticdem_resolution,
        #                 buffer=ceil(self.tpi_outer_radius / 2 * sqrt(2)),
        #             )
        #         with timer("Loading TCVis", log=False):
        #             tcvis = load_tcvis(tile.odc.geobox, self.tcvis_dir)
        #         with timer("Preprocessing tile", log=False):
        #             tile = preprocess_legacy_fast(
        #                 tile,
        #                 arcticdem,
        #                 tcvis,
        #                 self.tpi_outer_radius,
        #                 self.tpi_inner_radius,
        #                 self.device,
        #             )
        #         with timer("Segmenting", log=False):
        #             tile = ensemble.segment_tile(
        #                 tile,
        #                 patch_size=self.patch_size,
        #                 overlap=self.overlap,
        #                 batch_size=self.batch_size,
        #                 reflection=self.reflection,
        #                 keep_inputs=self.write_model_outputs,
        #             )
        #         with timer("Postprosessing", log=False):
        #             tile = prepare_export(
        #                 tile,
        #                 bin_threshold=self.binarization_threshold,
        #                 mask_erosion_size=self.mask_erosion_size,
        #                 min_object_size=self.min_object_size,
        #                 quality_level=self.quality_level,
        #                 ensemble_subsets=models.keys() if self.write_model_outputs else [],
        #                 device=self.device,
        #             )
        #
        #         with timer("Exporting", log=False):
        #             export_tile(
        #                 tile,
        #                 outpath,
        #                 bands=self.export_bands,
        #                 ensemble_subsets=models.keys() if self.write_model_outputs else [],
        #             )
        #
        #         n_tiles += 1
        #         results.append(
        #             {
        #                 "tile_id": tile_id,
        #                 "output_path": str(outpath.resolve()),
        #                 "status": "success",
        #                 "error": None,
        #             }
        #         )
        #         logger.info(f"Processed sample {i + 1} of {len(tileinfo)} '{tilekey}' ({tile_id=}).")
        #     except KeyboardInterrupt:
        #         logger.warning("Keyboard interrupt detected.\nExiting...")
        #         raise KeyboardInterrupt
        #     except Exception as e:
        #         logger.warning(f"Could not process '{tilekey}' ({tile_id=}).\nSkipping...")
        #         logger.exception(e)
        #         results.append(
        #             {
        #                 "tile_id": tile_id,
        #                 "output_path": str(outpath.resolve()),
        #                 "status": "failed",
        #                 "error": str(e),
        #             }
        #         )
        #     finally:
        #         if len(results) > 0:
        #             pd.DataFrame(results).to_parquet(self.output_data_dir / f"{current_time}.results.parquet")
        #         if len(timer.durations) > 0:
        #             timer.export().to_parquet(self.output_data_dir / f"{current_time}.stopuhr.parquet")
        else:
            logger.info(f"Processed {n_tiles} tiles to {self.output_data_dir.resolve()}.")
            timer.summary()


# =============================================================================
# Source Pipeliens
# =============================================================================
@dataclass
class PlanetPipelineRefactored(_BasePipelineRefactored):
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

    def _load_tile(self, fpath: Path) -> "xr.Dataset":
        import xarray as xr
        from darts_acquisition import load_planet_masks, load_planet_scene

        optical = load_planet_scene(fpath)
        data_masks = load_planet_masks(fpath)
        tile = xr.merge([optical, data_masks])
        return tile

    @staticmethod
    def cli(*, pipeline: "PlanetPipelineRefactored"):
        """Run the sequential pipeline for Planet data."""
        pipeline.run()


@dataclass
class Sentinel2PipelineRefactored(_BasePipelineRefactored):
    """Pipeline for Sentinel 2 data.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
            Defaults to Path("data/input/sentinel2").
        image_ids (list): The list of image ids to process. If None, all images in the directory will be processed.
            Defaults to None.

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

    sentinel2_dir: Path = Path("data/input/sentinel2")
    image_ids: list = None

    def _arcticdem_resolution(self) -> Literal[10]:
        return 10

    def _get_tile_id(self, tilekey: Path):
        from darts_acquisition import parse_s2_tile_id

        try:
            fpath = tilekey
            _, _, tile_id = parse_s2_tile_id(fpath)
            return tile_id
        except Exception as e:
            logger.error("Could not parse Sentinel 2 tile-id. Please check the input data.")
            logger.exception(e)
            raise e

    def _tileinfos(self) -> list[tuple[Path, Path]]:
        out = []
        for fpath in self.sentinel2_dir.glob("*/"):
            scene_id = fpath.name
            if self.image_ids is not None:
                if scene_id not in self.image_ids:
                    continue
            outpath = self.output_data_dir / scene_id
            out.append((fpath.resolve(), outpath))
        out.sort()
        return out

    def _load_tile(self, fpath: Path) -> "xr.Dataset":  # Here: fpath == 'tid'
        import xarray as xr
        from darts_acquisition import load_s2_masks, load_s2_scene

        optical = load_s2_scene(fpath)
        data_masks = load_s2_masks(fpath, optical.odc.geobox)
        tile = xr.merge([optical, data_masks])
        return tile

    @staticmethod
    def cli(*, pipeline: "Sentinel2PipelineRefactored"):
        """Run the sequential pipeline for Sentinel 2 data."""
        pipeline.run()


@dataclass
class AOISentinel2PipelineRefactored(_BasePipelineRefactored):
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
    input_cache: Path = Path("data/cache/input")

    def _arcticdem_resolution(self) -> Literal[10]:
        return 10

    @cached_property
    def _s2ids(self) -> list[str]:
        from darts_acquisition.s2 import get_s2ids_from_shape_ee

        return sorted(get_s2ids_from_shape_ee(self.aoi_shapefile, self.start_date, self.end_date, self.max_cloud_cover))

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

    def _load_tile(self, s2id: str) -> "xr.Dataset":
        from darts_acquisition.s2 import load_s2_from_gee

        tile = load_s2_from_gee(s2id, cache=self.input_cache)
        return tile

    @staticmethod
    def cli(*, pipeline: "AOISentinel2PipelineRefactored"):
        """Run the sequential pipeline for AOI Sentinel 2 data."""
        pipeline.run()
