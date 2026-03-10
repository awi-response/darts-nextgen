import pytest

from darts_export import vectorization


def test_regionstatsSimple(probabilities_1):
    polygon_gdf = vectorization.vectorize(probabilities_1)

    assert polygon_gdf.shape == (1, 8)
    p = polygon_gdf.iloc[0]

    assert p["min"] == 56
    assert p["max"] == 100
    assert p["mean"] == pytest.approx(76.0)
    assert p["median"] == pytest.approx(76.0)
    assert p["std"] == pytest.approx(15.38831)
    assert p.npixel == 20


def test_regionstatsMultiple(probabilities_2):
    polygon_gdf = vectorization.vectorize(probabilities_2)
    assert polygon_gdf.shape == (4, 8)

    polygon_gdf = vectorization.vectorize(probabilities_2)
    assert polygon_gdf.shape == (5, 8)
