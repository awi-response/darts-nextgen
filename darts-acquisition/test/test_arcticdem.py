import smart_geocubes
from numpy.testing import assert_almost_equal
from odc.geo.geobox import GeoBox
from pytest import approx

from darts_acquisition import load_arcticdem


def test_load_arcticdem():

    try:
        geobox = GeoBox.from_bbox((150, 65, 151, 65.5), shape=(1000, 1000))

        accessor = smart_geocubes.ArcticDEM32m("arcticdem_32m.icechunk", backend="threaded")
        if not accessor.created:
            accessor.create()
        accessor.assert_created()

        adem = load_arcticdem(
            geobox,
            data_dir="arcticdem_32m.icechunk",
            resolution=32,
            buffer=0,
            offline=False,
        )

        print(adem.dem.mean().item(), adem.dem.min().item(), adem.dem.max().item())
        print(f"{adem.dem.attrs=} {adem.arcticdem_data_mask.attrs=} {adem.attrs=}")
        assert adem.dem.mean().item() == approx(102.029907)
        assert adem.dem.min().item() == approx(46.34375)
        assert adem.dem.max().item() == approx(483.835937)
        assert_almost_equal(
            adem.odc.geobox.center_pixel.coords["x"].values / 1_000_000,
            geobox.to_crs("EPSG:3413").center_pixel.coords["x"].values / 1_000_000,
            decimal=4,
        )
        assert_almost_equal(
            adem.odc.geobox.center_pixel.coords["y"].values / 1_000_000,
            geobox.to_crs("EPSG:3413").center_pixel.coords["y"].values / 1_000_000,
            decimal=4,
        )

        # Check if dataset matches the expected structure and types
        assert {"dem", "arcticdem_data_mask"} == set(adem.data_vars)
        assert adem.dem.dtype == "float32"
        assert adem.arcticdem_data_mask.dtype == "uint8"
        assert set(adem.attrs.keys()) == {"title", "loaded_patches"}
        assert set(adem.dem.attrs.keys()) == {"units", "long_name", "data_source", "description"}
        assert set(adem.arcticdem_data_mask.attrs.keys()) == {"long_name", "data_source"}

    finally:
        if "adem" in locals():
            del adem
        # os.system("rm -rf arcticdem_32m.zarr")
