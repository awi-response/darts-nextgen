"""Major Tom grid implementation following the paper https://arxiv.org/html/2402.12095v2."""

import functools
import math
import re
from collections.abc import Generator
from dataclasses import asdict, dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely

R = 6378.137  # Earth radius at equator in km


def get_utm_zone_from_latlng(latitude: float | int, longitude: float | int) -> str:
    """Get the UTM zone from a latlng list and return the corresponding EPSG code.

    Args:
        latitude (float | int): The latitude of the point.
        longitude (float | int): The longitude of the point.

    Returns:
        str: The EPSG code corresponding to the UTM zone of the point.

    Raises:
        ValueError: If the point is out of bounds.

    """
    assert -180 <= longitude <= 180, f"longitude: {longitude} is out of bounds"
    assert -90 <= latitude <= 90, f"latitude: {latitude} is out of bounds"
    zone_number = (math.floor((longitude + 180) / 6)) % 60 + 1

    # Special zones for Svalbard and Norway
    if latitude >= 56.0 and latitude < 64.0 and longitude >= 3.0 and longitude < 12.0:
        zone_number = 32
    elif latitude >= 72.0 and latitude < 84.0:
        if longitude >= 0.0 and longitude < 9.0:
            zone_number = 31
        elif longitude >= 9.0 and longitude < 21.0:
            zone_number = 33
        elif longitude >= 21.0 and longitude < 33.0:
            zone_number = 35
        elif longitude >= 33.0 and longitude < 42.0:
            zone_number = 37

    # Determine the hemisphere and construct the EPSG code
    if latitude < 0:
        epsg_code = f"327{zone_number:02d}"
    else:
        epsg_code = f"326{zone_number:02d}"

    if not re.match(r"32[6-7](0[1-9]|[1-5][0-9]|60)", epsg_code):
        raise ValueError(
            f"The point {latitude=}, {longitude=} is out of bounds (resulted in {epsg_code=} and {zone_number=})"
        )

    return epsg_code


def parse_cell_idx(idx: str) -> tuple[int, str, int, str]:
    r"""Parse the Major Tom grid index into its components.

    Args:
        idx (str): Major Tom grid index in the format re'/\d+(D|U)\d+(L|R)/g'.

    Returns:
        tuple[int, str, int, str]: The row index, row direction, column index, and column direction.

    Raises:
        ValueError: If the index is invalid.

    """
    row_idx = ""
    row_direction = ""
    col_idx = ""
    col_direction = ""
    is_row = True
    for c in idx:
        if c.isdigit():
            if is_row:
                row_idx += c
            else:
                col_idx += c
        else:
            if is_row:
                row_direction = c
                is_row = False
            else:
                col_direction = c
    if not row_idx or not row_direction or not col_idx or not col_direction:
        raise ValueError(f"Invalid cell index: {idx}")
    return int(row_idx), row_direction, int(col_idx), col_direction


@dataclass(frozen=True)
class Cell:
    """Cell in the Major Tom grid."""

    d: float
    row_idx: int
    col_idx: int
    row_direction: Literal["U", "D"]
    col_direction: Literal["L", "R"]
    lat: float
    lon: float
    utm_zone: int

    @functools.cached_property
    def idx(self) -> str:
        """Major Tom grid index of the cell."""
        return f"{self.row_idx}{self.row_direction}{self.col_idx}{self.col_direction}"

    def __repr__(self) -> str:  # noqa: D105
        return f"Cell({self.idx})"

    def to_shape(self) -> shapely.Point:
        """Convert the cell to a shapely Point object.

        Returns:
            shapely.Point: The cell as a shapely Point object.

        """
        return shapely.geometry.Point(self.lon, self.lat)


class MajorTomGrid:
    """Major Tom grid following the implementation from https://arxiv.org/html/2402.12095v2."""

    def __init__(self, d: float):
        """Initialize the Major Tom grid with a given resolution.

        Args:
            d (float): Resolution of the grid in km.

        """
        self.d = d

    @functools.cached_property
    def n_r(self) -> int:
        """Number of rows (latitudes) in the grid."""
        return math.ceil(math.pi * R / self.d)

    @functools.cached_property
    def dlat(self) -> float:
        """Latitude resolution in degrees."""
        return 180 / self.n_r

    # Cache 2 calls so we can call n_c and dlon behind each other without doing nasty code
    @functools.lru_cache(maxsize=2)
    def n_c(self, lat: float) -> int:
        """Calculate the number of columns (longitudes) in the grid at a given latitude.

        Args:
            lat (float): Latitude at which to calculate the number of columns.

        Returns:
            int: Number of columns at the given latitude.

        """
        c_r = 2 * np.pi * R * math.cos(np.pi * lat / 180)  # Circumference of the circle at the given latitude
        return math.ceil(c_r / self.d)

    def dlon(self, lat: float) -> float:
        """Longitude resolution in degrees at a given latitude.

        Args:
            lat (float): Latitude at which to calculate the longitude resolution.

        Returns:
            float: Longitude resolution at the given latitude.

        """
        return 360 / self.n_c(lat)

    @functools.cache
    def __len__(self) -> int:  # noqa: D105
        n = 0
        for lat, _, _ in self.yield_latitudes():
            n += self.n_c(lat)
        return n

    def __getitem__(self, idx: str) -> Cell:
        r"""Get the grid cell at the given index.

        Args:
            idx (str): Major Tom grid index in the format re'/\d+(D|U)\d+(L|R)/g'.

        Returns:
            Cell: The grid cell at the given index.

        """
        row_idx, row_direction, col_idx, col_direction = parse_cell_idx(idx)

        # Get latitude information
        assert row_direction in ("U", "D")
        assert row_idx <= (self.n_r // 2)
        lat = row_idx * self.dlat * (-1 if row_direction == "D" else 1)

        # Get longitude information
        assert col_direction in ("L", "R")
        assert col_idx <= (self.n_c(lat) // 2)
        lon = col_idx * self.dlon(lat) * (-1 if col_direction == "L" else 1)

        utm_zone = get_utm_zone_from_latlng(lat, lon)
        return Cell(self.d, row_idx, col_idx, row_direction, col_direction, lat, lon, utm_zone)

    def yield_latitudes(self) -> Generator[tuple[float, int, str], None, None]:
        """Generate the latitudes of the Major Tom grid from north to south.

        Yields:
            tuple[float, int, str]: The next latitude in the grid. (latitude, row index, direction)

        """
        # if n_r is even -90 is included, 0 is always included

        # northern hemisphere
        northern_point = 90 - self.dlat / 2 if self.n_r % 2 else 90 - self.dlat
        n_northern = int(np.ceil(self.n_r / 2) - 1)
        for i, lat in enumerate(np.linspace(northern_point, 0, n_northern, endpoint=False)):
            yield lat, n_northern - i, "U"

        # equator
        yield 0, 0, "U"

        # southern hemisphere
        southern_point = -90 + self.dlat / 2 if self.n_r % 2 else -90
        n_southern = self.n_r // 2
        for i, lat in enumerate(np.linspace(-self.dlat, southern_point, n_southern, endpoint=True)):
            yield lat, i + 1, "D"

    def yield_longitudes(self, lat: float) -> Generator[tuple[float, int, str], None, None]:
        """Generate the longitudes of the Major Tom grid at a given latitude.

        Args:
            lat (float): Latitude at which to generate the longitudes.

        Yields:
            tuple[float, int, str]: The next longitude at the given latitude. (longitude, column index, direction)

        """
        n_c = self.n_c(lat)
        dlon = self.dlon(lat)

        # if n_c is even -180 is included, 0 is always included

        # western hemisphere
        western_point = -180 + dlon / 2 if n_c % 2 else -180
        n_western = n_c // 2
        for i, lon in enumerate(np.linspace(western_point, 0, n_western, endpoint=False)):
            yield lon, n_western - i, "L"

        # prime meridian
        yield 0, 0, "L"

        # eastern hemisphere
        eastern_point = 180 - dlon / 2 if n_c % 2 else 180 - dlon
        n_eastern = int(np.ceil(n_c / 2) - 1)
        for i, lon in enumerate(np.linspace(dlon, eastern_point, n_eastern, endpoint=True)):
            yield lon, i + 1, "R"

    def yield_cells(
        self,
        bounds: tuple[float, float, float, float] | shapely.Polygon | None = None,
    ) -> Generator[Cell, None, None]:
        """Generate all cells in the Major Tom grid.

        Starting at the North Pole, the generator yields cells from west to east and from north to south.

        Args:
            bounds (tuple[float, float, float, float] | shapely.Polygon | None): The bounds of the grid.
                If the bounds are a polygon, only cells within the polygon are yielded.
                If the bounds are a tuple, the format must be (min_lon, min_lat, max_lon, max_lat).
                If None, the entire grid is yielded.
                Coordinates must be in EPSG:4326.
                Please note that bounds around the antimetidian (180°/-180°) and the poles (90°/-90°) are not supported.
                Defaults to None.

        Yields:
            Cell: The next cell in the grid.

        """
        is_polygon = isinstance(bounds, shapely.Polygon)
        if is_polygon:
            orig_min_lon, min_lat, max_lon, max_lat = bounds.bounds
        else:
            orig_min_lon, min_lat, max_lon, max_lat = bounds or (-180, -90, 180, 90)
        # Include grid points just outside the bounds to ensure that their cell (to the north-west) is included
        min_lat -= self.dlat

        for lat, row_idx, row_direction in self.yield_latitudes():
            # Include grid points just outside the bounds to ensure that their cell (to the north-west) is included
            min_lon = orig_min_lon - self.dlon(lat)
            for lon, col_idx, col_direction in self.yield_longitudes(lat):
                # Do an easy-bound check first
                if not (min_lat <= lat <= max_lat) or not (min_lon <= lon <= max_lon):
                    continue
                # If the bounds are a polygon, do a more precise check
                if (
                    is_polygon
                    and not bounds.contains(shapely.geometry.Point(lon, lat))
                    and not bounds.contains(shapely.geometry.Point(lon + self.dlon(lat), lat))
                    and not bounds.contains(shapely.geometry.Point(lon, lat + self.dlat))
                    and not bounds.contains(shapely.geometry.Point(lon + self.dlon(lat), lat + self.dlat))
                ):
                    continue
                utm_zone = get_utm_zone_from_latlng(lat, lon)
                yield Cell(self.d, row_idx, col_idx, row_direction, col_direction, lat, lon, utm_zone)

    def to_geodataframe(
        self, bounds: tuple[float, float, float, float] | shapely.Polygon | None = None
    ) -> gpd.GeoDataFrame:
        """Convert the Major Tom grid to a geopandas GeoDataFrame.

        Args:
            bounds (tuple[float, float, float, float] | shapely.Polygon | None): The bounds of the grid.
                If the bounds are a polygon, only cells within the polygon are yielded.
                If the bounds are a tuple, the format must be (min_lon, min_lat, max_lon, max_lat).
                If None, the entire grid is yielded.
                Coordinates must be in EPSG:4326.
                Defaults to None.

        Returns:
            geopandas.GeoDataFrame: The Major Tom grid as a GeoDataFrame.

        """
        cells = [{"geometry": shapely.Point([cell.lon, cell.lat]), **asdict(cell)} for cell in self.yield_cells(bounds)]
        gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")
        return gdf
