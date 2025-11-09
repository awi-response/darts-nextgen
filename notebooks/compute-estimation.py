# ruff: noqa: D101, D102, D103
"""Simple estimation of panarctic permafrost Sentinel 2 tiles."""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

import cyclopts
import geopandas as gpd
from pyproj import CRS
from pystac_client import Client
from rich import pretty, traceback
from rich.logging import RichHandler
from rich.progress import track
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

traceback.install(show_locals=True, suppress=[cyclopts])
pretty.install()

out = Path("/isipd/projects/p_aicore_pf/darts-nextgen/data")
out.mkdir(parents=True, exist_ok=True)
stac_cache = out / "s2permaice.stac"
stac_cache.mkdir(parents=True, exist_ok=True)


download_lock = Lock()
last_request = time.time()


def check_request_limit():
    global last_request

    # Ensure at least tbr (time between requests) second between requests
    tbr = 3
    with download_lock:
        elapsed = time.time() - last_request
        if elapsed < tbr:
            time.sleep(tbr - elapsed)
        last_request = time.time()


catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")


def load_grid_ids() -> set[str]:
    # MGRS Grid from https://github.com/DPIRD-DMA/Sentinel-2-grid-explorer/tree/main
    mgrs = gpd.read_file("./data/sentinel-2_grids.geojson")

    # Permafrost extent from https://nsidc.org/data/ggd318/versions/2
    permaice = gpd.read_file("./data/Permafrost/permaice.shp")
    permaice = permaice[~permaice["CONTENT"].isna() & (permaice["COMBO"] != "o")]

    # Get mgrs IDs based on permaice extent
    mgrs_permaice = mgrs[mgrs.to_crs(permaice.crs).intersects(permaice.geometry.union_all(method="coverage"))]

    # Filter out all MGRS tiles which are below 55 degree
    mgrs_permaice = mgrs_permaice[mgrs_permaice.geometry.bounds.miny >= 55]

    return set(mgrs_permaice["Name"].to_list())


@dataclass
class S2Stats:
    id: str
    snow_coverage: float
    cloud_coverage: float
    water_coverage: float
    nodata: float
    date: datetime
    area: float
    utm_crs: str
    grid_code: str
    geometry: Polygon

    @classmethod
    def from_item(cls, row: gpd.GeoSeries, utm_crs: CRS) -> "S2Stats":
        if isinstance(row["statistics"], dict):
            water_coverage = row.statistics["water"]
            nodata = row.statistics["nodata"]
        else:
            water_coverage = float("nan")
            nodata = float("nan")

        return S2Stats(
            id=row["id"],
            snow_coverage=row["eo:snow_cover"],
            cloud_coverage=row["eo:cloud_cover"],
            water_coverage=water_coverage,
            nodata=nodata,
            date=datetime.strptime(row.datetime, "%Y-%m-%dT%H:%M:%S.%fZ"),
            area=row.area,
            utm_crs=utm_crs.to_epsg(),
            grid_code=row["grid:code"].split("-")[1],
            geometry=row.geometry,
        )


def download_stac(gid: str):
    check_request_limit()
    logger.debug(f"Searching S2 items for grid '{gid}'")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        query=[f"grid:code=MGRS-{gid}", "eo:cloud_cover<=40"],
    )
    s2items = search.item_collection().to_dict()
    (stac_cache / f"s2stats-{gid}.json").write_text(json.dumps(s2items))


def estimate(grid_code: str) -> list[S2Stats]:
    s2items_fpath = stac_cache / f"s2stats-{grid_code}.json"
    try:
        s2items = json.loads(s2items_fpath.read_text())
        assert len(s2items["features"]) > 0
    except (KeyboardInterrupt, SystemError, SystemExit) as e:
        raise e
    except Exception:
        logger.debug(f"No S2 tiles found for grid '{grid_code}'")
        return []

    s2gdf = gpd.GeoDataFrame.from_features(s2items).set_crs("EPSG:4326")
    utm_crs = CRS.from_string(f"+proj=utm +zone={grid_code[:2]} +north")
    s2gdf["area"] = s2gdf.to_crs(utm_crs).geometry.area / 1e6  # convert to km2
    s2gdf["id"] = [item["id"] for item in s2items["features"]]

    stats = [S2Stats.from_item(row, utm_crs) for _, row in s2gdf.iterrows()]
    return stats


def cli(threads: int = 5, workers: int = 4, download: bool = False):  # noqa: C901
    """Download metadata information from the Sentinel 2 STAC API and calculate simple statistics from them.

    Args:
        threads (int, optional): Number of threads to use for the download. Defaults to 5.
        workers (int, optional): Number of workers to use for the computation of simplified statistics. Defaults to 4.
        download (bool, optional): Whether to download missing STAC items. Defaults to False.

    Raises:
        KeyboardInterrupt: If user interrupts execution.
        SystemExit: If the process is terminated.
        SystemError: If a system error occurs.

    """
    grid_ids = load_grid_ids()
    # Download missing STAC items
    grid_ids_to_download = [gid for gid in grid_ids if not (stac_cache / f"s2stats-{gid}.json").exists()]
    if len(grid_ids_to_download) > 0 and download:
        logger.info(f"Downloading STAC items for {len(grid_ids_to_download)} grid tiles")
        n_err = 0
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(download_stac, gid): gid for gid in grid_ids_to_download}
            for future in track(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except (KeyboardInterrupt, SystemError, SystemExit) as e:
                    logger.warning(f"{type(e).__name__} detected.\nExiting...")
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise e
                except Exception:
                    n_err += 1
        if n_err:
            logger.error(f"Failed downloading {n_err} tiles")
            return
        logger.info("Finished downloading STAC items")

    n_err = 0
    s2stats = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for grid_id in grid_ids:
            futures[executor.submit(estimate, grid_id)] = grid_id

        logger.info(f"Processing {len(futures)} grid tiles")

        for future in track(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if not result:
                    logger.info(f"Processed '{futures[future]}': no tiles found")
                    continue
                grid_id = futures[future]
                logger.info(f"Processed '{grid_id}': found {len(result)} tiles")
                s2stats.extend(result)
            except KeyboardInterrupt as e:
                executor.shutdown(wait=True, cancel_futures=True)
                raise e
            except Exception:
                n_err += 1
                # logger.error(f"Error processing grid '{futures[future]}'", exc_info=e)

    if n_err:
        logger.info(f"Failed for {n_err} tiles")

    if len(s2stats) == 0:
        logger.info("No valid S2 statistics found.")
        return

    s2stats_gdf = gpd.GeoDataFrame([asdict(stat) for stat in s2stats]).set_crs("EPSG:4326")
    s2stats_gdf.to_file(out / "s2stats.geojson")
    s2stats_gdf.to_parquet(out / "s2stats.parquet")

    logger.info(f"Saved {len(s2stats)} S2 stats")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(
        RichHandler(
            rich_tracebacks=True, tracebacks_show_locals=True, tracebacks_suppress=[cyclopts], level=logging.DEBUG
        )
    )
    cyclopts.run(cli)
