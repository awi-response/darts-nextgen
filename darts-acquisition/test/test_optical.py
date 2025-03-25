import os

import ee
import xarray as xr

from darts_acquisition.s2 import load_s2_from_gee, load_s2_from_stac, load_s2_masks, load_s2_scene


def assert_optical_dataset(tile: xr.Dataset):
    # Check if the dataset contains the correct variables
    assert "red" in tile.data_vars
    assert "green" in tile.data_vars
    assert "blue" in tile.data_vars
    assert "nir" in tile.data_vars
    assert "quality_data_mask" in tile.data_vars

    # Check if the dataset contains the correct dimensions
    assert "x" in tile.dims
    assert "y" in tile.dims

    # Check if the dataset contains the correct datatypes
    assert tile.red.dtype == "uint16"
    assert tile.green.dtype == "uint16"
    assert tile.blue.dtype == "uint16"
    assert tile.nir.dtype == "uint16"
    assert tile.quality_data_mask.dtype == "uint8"

    # Check for right georefence
    assert tile.odc.geobox is not None
    # TODO: More check for georefence

    # Check for necessary attributes
    assert "tile_id" in tile.attrs

    for band in ["red", "green", "blue", "nir"]:
        assert "data_source" in tile[band].attrs, f"data_source not found in {band=}"
        assert "long_name" in tile[band].attrs, f"long_name not found in {band=}"
        assert "units" in tile[band].attrs, f"units not found in {band=}"

    assert "data_source" in tile["quality_data_mask"].attrs
    assert "long_name" in tile["quality_data_mask"].attrs


def test_load_s2_scene():
    fpath = "./data/input/sentinel2/20210818T223529_20210818T223531_T03WXP"
    tile = load_s2_scene(fpath)
    mask = load_s2_masks(fpath, tile.odc.geobox)
    tile = xr.merge([tile, mask])
    assert_optical_dataset(tile)


def test_load_s2_from_gee():
    ee_project = os.environ.get("EE_PROJECT")
    ee.Authenticate()
    ee.Initialize(project=ee_project)
    s2id = "20240614T211521_20240614T211517_T08WMA"
    tile = load_s2_from_gee(s2id)
    assert_optical_dataset(tile)


def test_load_s2_from_stac():
    s2id = "S2B_MSIL2A_20250306T125309_N0511_R138_T27WXN_20250306T133359"
    tile = load_s2_from_stac(s2id)
    assert_optical_dataset(tile)
