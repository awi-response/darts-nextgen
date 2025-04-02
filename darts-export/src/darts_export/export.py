"""Darts export module for inference results."""

import logging
from pathlib import Path

import stopuhr
import xarray as xr

from darts_export import miniviz, vectorization

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _export_raster(tile: xr.Dataset, name: str, out_dir: Path, fname: str | None = None):
    if fname is None:
        fname = name
    fpath = out_dir / f"{fname}.tif"
    with stopuhr.stopuhr(f"Exporting {name} to {fpath.resolve()}", logger.debug):
        tile[name].rio.to_raster(fpath, driver="GTiff", compress="LZW")


def _export_vector(tile: xr.Dataset, name: str, out_dir: Path, fname: str | None = None):
    if fname is None:
        fname = name
    fpath_gpkg = out_dir / f"{fname}.gpkg"
    fpath_parquet = out_dir / f"{fname}.parquet"
    with stopuhr.stopuhr(f"Exporting {name} to {fpath_gpkg.resolve()} and {fpath_parquet.resolve()}", logger.debug):
        polygon_gdf = vectorization.vectorize(tile, name)
        polygon_gdf.to_file(fpath_gpkg, layer=f"{fname}")
        polygon_gdf.to_parquet(fpath_parquet)


def _export_polygonized(tile: xr.Dataset, out_dir: Path, ensemble_subsets: list[str] = []):
    _export_vector(tile, "binarized_segmentation", out_dir, fname="prediction_segments")
    for ensemble_subset in ensemble_subsets:
        _export_vector(
            tile,
            f"binarized_segmentation-{ensemble_subset}",
            out_dir,
            fname=f"prediction_segments-{ensemble_subset}",
        )


def _export_binarized(tile: xr.Dataset, out_dir: Path, ensemble_subsets: list[str] = []):
    _export_raster(tile, "binarized_segmentation", out_dir, fname="binarized")
    for ensemble_subset in ensemble_subsets:
        _export_raster(
            tile,
            f"binarized_segmentation-{ensemble_subset}",
            out_dir,
            fname=f"binarized-{ensemble_subset}",
        )


def _export_probabilities(tile: xr.Dataset, out_dir: Path, ensemble_subsets: list[str] = []):
    _export_raster(tile, "probabilities", out_dir, fname="probabilities")
    for ensemble_subset in ensemble_subsets:
        _export_raster(
            tile,
            f"probabilities-{ensemble_subset}",
            out_dir,
            fname=f"probabilities-{ensemble_subset}",
        )


def _export_thumbnail(tile: xr.Dataset, out_dir: Path):
    fpath = out_dir / "thumbnail.jpg"
    with stopuhr.stopuhr(f"Exporting thumbnail to {fpath}", logger.debug):
        fig = miniviz.thumbnail(tile)
        fig.savefig(fpath)
        fig.clear()


@stopuhr.funkuhr("Exporting tile", logger.debug, print_kwargs=["bands", "ensemble_subsets"])
def export_tile(  # noqa: C901
    tile: xr.Dataset,
    out_dir: Path,
    bands: list[str] = ["probabilities", "binarized", "polygonized", "extent", "thumbnail"],
    ensemble_subsets: list[str] = [],
):
    """Export a tile to a file.

    Args:
        tile (xr.Dataset): The tile to export.
        out_dir (Path): The path where to export to.
        bands (list[str], optional): The bands to export. Defaults to ["probabilities"].
        ensemble_subsets (list[str], optional): The ensemble subsets to export. Defaults to [].

    Raises:
        ValueError: If the band is not found in the tile.

    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for band in bands:
        match band:
            case "polygonized":
                _export_polygonized(tile, out_dir, ensemble_subsets)
            case "binarized":
                _export_binarized(tile, out_dir, ensemble_subsets)
            case "probabilities":
                _export_probabilities(tile, out_dir, ensemble_subsets)
            case "extent":
                _export_vector(tile, "extent", out_dir, fname="prediction_extent")
            case "thumbnail":
                _export_thumbnail(tile, out_dir)
            case "optical":
                _export_raster(tile, ["red", "green", "blue", "nir"], out_dir, fname="optical")
            case "dem":
                _export_raster(tile, ["slope", "relative_elevation"], out_dir, fname="dem")
            case "tcvis":
                _export_raster(tile, ["tc_brightness", "tc_greenness", "tc_wetness"], out_dir, fname="tcvis")
            case _:
                if band not in tile.data_vars:
                    raise ValueError(
                        f"Band {band} not found in tile for export. Available bands are: {list(tile.data_vars.keys())}"
                    )
                # Export the band as a raster
                _export_raster(tile, band, out_dir)
