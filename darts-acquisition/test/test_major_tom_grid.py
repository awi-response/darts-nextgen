import math

import numpy as np
import pytest

from darts_acquisition.utils.grid import MajorTomGrid

# Generate numbers between 1 and 1000 on a logarithmic scale
d_vals = [1, 3, 10]
# d_vals = np.logspace(0.1, 3, num=100
#   -> This fails because the original implementation sometime use the north pole and
#      sometime the south pole as the starting point, we use always the south pole

R = 6378.137  # Earth radius at equator in km


@pytest.mark.parametrize("d", d_vals)
def test_latitudes(d: float):
    # Calculate the latitudes of the orginial implementation
    arc_pole_to_pole = math.pi * R
    num_divisions_in_hemisphere = math.ceil(arc_pole_to_pole / d)
    latitudes = np.linspace(-90, 90, num_divisions_in_hemisphere + 1)[:-1]
    latitudes = np.mod(latitudes, 180) - 90
    latitudes = np.sort(latitudes)

    # From 0U-NU and 1D-ND
    zeroth_row = np.searchsorted(latitudes, 0)
    rows = [None] * len(latitudes)
    rows[zeroth_row:] = [f"{i}U" for i in range(len(latitudes) - zeroth_row)]
    rows[:zeroth_row] = [f"{abs(i - zeroth_row)}D" for i in range(zeroth_row)]

    # Calculate the latitudes of the new implementation
    grid = MajorTomGrid(d)
    latitudes_new = np.array([cell[0] for cell in grid.yield_latitudes()])[::-1]
    rows_new = [f"{cell[1]}{cell[2]}" for cell in grid.yield_latitudes()][::-1]

    # Check if the latitudes are the same
    np.testing.assert_allclose(latitudes, latitudes_new, atol=1e-8)

    # Check if the rows are the same
    for row, row_new in zip(rows, rows_new):
        assert row == row_new


@pytest.mark.parametrize("d", d_vals)
def test_validity(d: float):
    grid = MajorTomGrid(d)
    for cell in grid.yield_cells():
        assert cell.row_idx >= 0
        assert cell.col_idx >= 0
        assert cell.lat >= -90
        assert cell.lat < 90
        assert cell.lon >= -180
        assert cell.lon <= 180
