"""Parallel implementation of the v2 pipelines using Ray."""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import cached_property
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import ray

from cyclopts import Parameter

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


@Parameter(name="*")
@dataclass
class BasePipelineRay(ABC):
    """Base class for all v2 pipelines with Ray parallel processing."""

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
    num_workers: int = 1  # Number of parallel workers

    @abstractmethod
    def _arcticdem_resolution(self) -> Literal[2, 10, 32]:
        """Return the resolution of the ArcticDEM data."""
        pass

    @abstractmethod
    def _get_tile_id(self, tilekey: Any) -> str:
        pass

    @abstractmethod
    def _tileinfos(self) -> list[tuple[Any, Path]]:
        pass

    @abstractmethod
    def _load_tile(self, tileinfo: Any) -> "xr.Dataset":
        pass

    @ray.remote
    def _process_tile(self, i, tilekey, outpath, current_time, config_dict):
        """Ray remote function to process a single tile."""
        # Reconstruct necessary objects from config
        from stopuhr import Chronometer
        import torch
        from darts_ensemble import EnsembleV1
        from darts_acquisition import load_arcticdem, load_tcvis
        from darts_export import export_tile, missing_outputs
        from darts_postprocessing import prepare_export
        from darts_preprocessing import preprocess_legacy_fast
        import pandas as pd

        # Create local timer and results
        timer = Chronometer(printer=logger.debug)
        results = []
        n_tiles = 0

        # Recreate models and ensemble
        models = {Path(k).stem: Path(k) for k in config_dict["model_files"]}
        ensemble = EnsembleV1(models, device=torch.device(config_dict["device"]))

        tile_id = self._get_tile_id(tilekey)
        result = {
            "tile_id": tile_id,
            "output_path": str(outpath.resolve()),
            "status": "failed",
            "error": None,
        }

        try:
            if not config_dict["overwrite"]:
                mo = missing_outputs(outpath, bands=config_dict["export_bands"],
                                     ensemble_subsets=models.keys())
                if mo == "none":
                    result["status"] = "skipped"
                    return ([result], 0)
                if mo == "some":
                    result["status"] = "skipped_partial"
                    return ([result], 0)

            with timer("Loading optical data", log=False):
                tile = self._load_tile(tilekey)

            with timer("Loading ArcticDEM", log=False):
                arcticdem = load_arcticdem(
                    tile.odc.geobox,
                    Path(config_dict["arcticdem_dir"]),
                    resolution=self._arcticdem_resolution(),
                    buffer=ceil(config_dict["tpi_outer_radius"] / 2 * sqrt(2)),
                )

            with timer("Loading TCVis", log=False):
                tcvis = load_tcvis(tile.odc.geobox, Path(config_dict["tcvis_dir"]))

            with timer("Preprocessing tile", log=False):
                tile = preprocess_legacy_fast(
                    tile,
                    arcticdem,
                    tcvis,
                    config_dict["tpi_outer_radius"],
                    config_dict["tpi_inner_radius"],
                    config_dict["device"],
                )

            with timer("Segmenting", log=False):
                tile = ensemble.segment_tile(
                    tile,
                    patch_size=config_dict["patch_size"],
                    overlap=config_dict["overlap"],
                    batch_size=config_dict["batch_size"],
                    reflection=config_dict["reflection"],
                    keep_inputs=config_dict["write_model_outputs"],
                )

            with timer("Postprocessing", log=False):
                tile = prepare_export(
                    tile,
                    bin_threshold=config_dict["binarization_threshold"],
                    mask_erosion_size=config_dict["mask_erosion_size"],
                    min_object_size=config_dict["min_object_size"],
                    quality_level=config_dict["quality_level"],
                    ensemble_subsets=models.keys() if config_dict["write_model_outputs"] else [],
                    device=config_dict["device"],
                )

            with timer("Exporting", log=False):
                export_tile(
                    tile,
                    outpath,
                    bands=config_dict["export_bands"],
                    ensemble_subsets=models.keys() if config_dict["write_model_outputs"] else [],
                )

            results.append({
                "tile_id": tile_id,
                "output_path": str(outpath.resolve()),
                "status": "success",
                "error": None,
            })
            n_tiles = 1

        except Exception as e:
            logger.warning(f"Could not process '{tilekey}' ({tile_id=}).\nSkipping...")
            results.append({
                "tile_id": tile_id,
                "output_path": str(outpath.resolve()),
                "status": "failed",
                "error": str(e),
            })

        return (results, n_tiles)

    def run(self):
        if self.model_files is None or len(self.model_files) == 0:
            raise ValueError("No model files provided.")
        if len(self.export_bands) == 0:
            raise ValueError("No export bands provided.")

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"Starting pipeline at {current_time} with {self.num_workers} workers.")

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_workers)

        # Prepare configuration dictionary
        config_dict = asdict(self)
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value.resolve())
            elif isinstance(value, list) and value and isinstance(value[0], Path):
                config_dict[key] = [str(v.resolve()) for v in value]

        # Get tile information
        tileinfo = self._tileinfos()
        logger.info(f"Found {len(tileinfo)} tiles to process.")

        # Process tiles in parallel
        futures = []
        for i, (tilekey, outpath) in enumerate(tileinfo):
            futures.append(
                self._process_tile.remote(self, i, tilekey, outpath, current_time, config_dict)
            )

        # Collect results
        all_results = []
        total_n_tiles = 0
        while futures:
            done, futures = ray.wait(futures)
            for result in ray.get(done):
                results, n_tiles = result
                all_results.extend(results)
                total_n_tiles += n_tiles

                # Save intermediate results
                if all_results:
                    import pandas as pd
                    pd.DataFrame(all_results).to_parquet(
                        self.output_data_dir / f"{current_time}.results.parquet"
                    )

        logger.info(f"Processed {total_n_tiles} tiles to {self.output_data_dir.resolve()}.")


@dataclass
class AOISentinel2PipelineRay(BasePipelineRay):
    """Ray-parallel pipeline for Sentinel 2 data based on an area of interest."""

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
        return sorted(get_s2ids_from_shape_ee(
            self.aoi_shapefile,
            self.start_date,
            self.end_date,
            self.max_cloud_cover
        ))

    def _get_tile_id(self, tilekey):
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
        return load_s2_from_gee(s2id, cache=self.input_cache)

    @staticmethod
    def cli(*, pipeline: "AOISentinel2PipelineRay"):
        """Run the parallel pipeline for AOI Sentinel 2 data."""
        pipeline.run()