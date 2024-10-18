import numpy as np
import pytest
from skimage import measure

from darts_export import vectorization as vect

vectfunc_params = [
    pytest.param(vect.rasterio_polygonization, id="rasterio"),
    pytest.param(vect.gdal_polygonization, id="gdal"),
]


@pytest.mark.parametrize("vector_func", vectfunc_params)
def test_vectorizationSimple(probabilities_1: np.ndarray, vector_func):
    layer = probabilities_1.binarized_segmentation

    bin_labelled = measure.label(layer)

    gdf_polygons = vector_func(bin_labelled, layer)

    assert gdf_polygons.shape == (1, 2)
    assert max(gdf_polygons.Region_ID) == 1


@pytest.mark.parametrize("vector_func", vectfunc_params)
def test_vectorizationMultiple(probabilities_2: np.ndarray, vector_func):
    layer = probabilities_2.binarized_segmentation
    bin_labelled = measure.label(layer)

    gdf_polygons = vector_func(bin_labelled, layer)

    assert gdf_polygons.shape == (5, 2)
    assert max(gdf_polygons.Region_ID) == 5

    # check if the largest polygon has this shape
    expected_geometry_blob = (
        "POLYGON (("
        "500076.5 4500004.5, 500076.5 4500005.5, 500075.5 4500005.5, 500075.5 4500006.5, "
        "500074.5 4500006.5, 500074.5 4500017.5, 500073.5 4500017.5, 500073.5 4500018.5, "
        "500072.5 4500018.5, 500072.5 4500019.5, 500056.5 4500019.5, 500056.5 4500020.5, "
        "500055.5 4500020.5, 500055.5 4500021.5, 500054.5 4500021.5, 500054.5 4500028.5, "
        "500055.5 4500028.5, 500055.5 4500029.5, 500056.5 4500029.5, 500056.5 4500030.5, "
        "500075.5 4500030.5, 500075.5 4500029.5, 500076.5 4500029.5, 500076.5 4500028.5, "
        "500077.5 4500028.5, 500077.5 4500024.5, 500078.5 4500024.5, 500078.5 4500023.5, "
        "500079.5 4500023.5, 500079.5 4500022.5, 500082.5 4500022.5, 500082.5 4500021.5, "
        "500083.5 4500021.5, 500083.5 4500020.5, 500084.5 4500020.5, 500084.5 4500015.5, "
        "500085.5 4500015.5, 500085.5 4500010.5, 500084.5 4500010.5, 500084.5 4500006.5, "
        "500083.5 4500006.5, 500083.5 4500005.5, 500082.5 4500005.5, 500082.5 4500004.5, "
        "500076.5 4500004.5"
        "))"
    )

    # largest polygon:
    top_poly = gdf_polygons[max(gdf_polygons.geometry.area) == gdf_polygons.geometry.area].iloc[0]
    assert str(top_poly.geometry) == expected_geometry_blob
