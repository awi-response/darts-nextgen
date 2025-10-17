import os
from pathlib import Path
from typing import Literal

import ee
import numpy as np
import pytest
import xarray as xr

from darts_acquisition import load_cdse_s2_sr_scene, load_gee_s2_sr_scene, load_planet_masks, load_planet_scene

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve() / "darts-unit-tests"

GEE_IMAGES = [
    "20210818T223529_20210818T223531_T03WXP",
    "20200726T071621_20200726T072016_T42WWA",
    "20240725T071621_20240725T071928_T42WWA",
]

STAC_IMAGES = [
    "S2B_MSIL2A_20210818T223529_N0500_R058_T03WXP_20230119T164502",
    "S2A_MSIL2A_20200726T071621_N0500_R006_T42WWA_20230505T194826",
    "S2A_MSIL2A_20240725T071621_N0511_R006_T42WWA_20240725T122945",
]

PLANET_SCENES = [
    "PSOrthoTile/4372514/5790392_4372514_2022-07-16_2459",
    "PSOrthoTile/4974017/5854937_4974017_2022-08-14_2475",
    "PSScene/20230703_194241_43_2427",
    "PSScene/20230703_194243_54_2427",
]

S2_STATS = {
    # stac only - values derived from gee -> fix it
    "S2B_MSIL2A_20210818T223529_N0500_R058_T03WXP_20230119T164502": {
        "blue": {
            "min": -0.00010000169277191162,
            "max": 1.3055999279022217,
            "mean": 0.09508587419986725,
            "std": 0.1749054193496704,
        },
        "green": {
            "min": -0.0007999986410140991,
            "max": 1.3087999820709229,
            "mean": 0.10263977199792862,
            "std": 0.16013139486312866,
        },
        "red": {
            "min": 0.001099996268749237,
            "max": 1.3335999250411987,
            "mean": 0.09263268858194351,
            "std": 0.15675930678844452,
        },
        "nir": {
            "min": -0.029600001871585846,
            "max": 1.5576000213623047,
            "mean": 0.2068384438753128,
            "std": 0.2032119333744049,
        },
    },
    "S2A_MSIL2A_20200726T071621_N0500_R006_T42WWA_20230505T194826": {
        "blue": {
            "min": -0.09989999979734421,
            "max": 0.9104000329971313,
            "mean": 0.05048460140824318,
            "std": 0.0595678985118866,
        },
        "green": {
            "min": -0.07519999891519547,
            "max": 0.8791999816894531,
            "mean": 0.07287579029798508,
            "std": 0.05911505967378616,
        },
        "red": {
            "min": -0.0394,
            "max": 0.8791999816894531,
            "mean": 0.07206938415765762,
            "std": 0.06203201413154602,
        },
        "nir": {
            "min": -0.030900001525878906,
            "max": 0.9855999946594238,
            "mean": 0.2371862232685089,
            "std": 0.1223120465874672,
        },
    },
    # gee only
    "S2B_MSIL2A_20210818T223529_N0301_R058_T03WXP_20210818T232618": {
        "blue": {"max": 1.2912, "mean": 0.09158431310431633, "min": 0.0001, "std": 0.17380070822417765},
        "green": {"max": 1.2248, "mean": 0.09998463102825343, "min": 0.0001, "std": 0.15911134390543405},
        "red": {"max": 1.208, "mean": 0.09068416482992385, "min": 0.0001, "std": 0.15560658378365516},
        "nir": {"max": 1.4288, "mean": 0.203464148073712, "min": 0.0001, "std": 0.2003786383085887},
    },
    "S2A_MSIL2A_20200726T071621_N0214_R006_T42WWA_20200726T100056": {
        "blue": {"max": 0.9152, "mean": 0.0509269718387613, "min": 0.0001, "std": 0.05962125470515183},
        "green": {"max": 0.8904, "mean": 0.07406982288177263, "min": 0.0001, "std": 0.059546196510380456},
        "red": {"max": 0.8864, "mean": 0.07241611037452454, "min": 0.0001, "std": 0.06213890719216461},
        "nir": {"max": 0.9856, "mean": 0.23640621645855237, "min": 0.0001, "std": 0.12143236055584375},
    },
    # Both gee and stac
    "S2A_MSIL2A_20240725T071621_N0511_R006_T42WWA_20240725T122945": {
        "blue": {"max": 0.7424, "mean": 0.047936902106954435, "min": -0.02690, "std": 0.054840356824684564},
        "green": {"max": 0.714, "mean": 0.07024756402654031, "min": -0.0109, "std": 0.053981019745416504},
        "red": {"max": 0.712, "mean": 0.07072753553760988, "min": -0.0084, "std": 0.05675720501322804},
        "nir": {"max": 0.7976, "mean": 0.23647443124171064, "min": -0.0302, "std": 0.11491599581770419},
    },
}

PLANET_STATS = {
    "4372514-5790392_4372514_2022-07-16_2459": {
        "blue": {"min": 0.0, "max": 0.09019999951124191, "mean": 0.000878745864611119, "std": 0.004382252227514982},
        "green": {"min": 0.0, "max": 0.09870000183582306, "mean": 0.001503659412264824, "std": 0.007427905220538378},
        "red": {"min": 0.0, "max": 0.09449999779462814, "mean": 0.0014526360901072621, "std": 0.007178109139204025},
        "nir": {"min": 0.0, "max": 0.07720000296831131, "mean": 0.0005429781740531325, "std": 0.0027963200118392706},
    },
    "4974017-5854937_4974017_2022-08-14_2475": {
        "blue": {"min": 0.0, "max": 0.5866000056266785, "mean": 0.032545749098062515, "std": 0.01955529861152172},
        "green": {"min": 0.0, "max": 0.5631999969482422, "mean": 0.048861514776945114, "std": 0.027387026697397232},
        "red": {"min": 0.0, "max": 0.6413999795913696, "mean": 0.04844691976904869, "std": 0.029639501124620438},
        "nir": {"min": 0.0, "max": 0.6370999813079834, "mean": 0.11692561209201813, "std": 0.1262866109609604},
    },
    "20230703_194241_43_2427": {
        "blue": {
            "min": 0.009600000455975533,
            "max": 0.32429999113082886,
            "mean": 0.03120855800807476,
            "std": 0.010105404071509838,
        },
        "green": {
            "min": 0.014000000432133675,
            "max": 0.38429999351501465,
            "mean": 0.0495685450732708,
            "std": 0.016076313331723213,
        },
        "red": {
            "min": 0.007199999876320362,
            "max": 0.4115000069141388,
            "mean": 0.03678949549794197,
            "std": 0.019877061247825623,
        },
        "nir": {
            "min": 9.999999747378752e-05,
            "max": 0.7024999856948853,
            "mean": 0.22623081505298615,
            "std": 0.12397556006908417,
        },
    },
    "20230703_194243_54_2427": {
        "blue": {
            "min": 0.007400000002235174,
            "max": 0.328000009059906,
            "mean": 0.031715378165245056,
            "std": 0.011518585495650768,
        },
        "green": {
            "min": 0.012900000438094139,
            "max": 0.37880000472068787,
            "mean": 0.05263156071305275,
            "std": 0.018013842403888702,
        },
        "red": {
            "min": 0.007899999618530273,
            "max": 0.3824000060558319,
            "mean": 0.04049612954258919,
            "std": 0.02256166562438011,
        },
        "nir": {
            "min": 0.0003000000142492354,
            "max": 0.7032999992370605,
            "mean": 0.2537764310836792,
            "std": 0.118050716817379,
        },
    },
}


def assert_optical_dataset(tile: xr.Dataset, satellite: Literal["s2", "planet"]):
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
    assert tile.red.dtype == "float32"
    assert tile.green.dtype == "float32"
    assert tile.blue.dtype == "float32"
    assert tile.nir.dtype == "float32"
    assert tile.quality_data_mask.dtype == "uint8"

    # TODO: Test for presence of encoding / compression information which could fuck up saving

    # Check for right georefence
    assert tile.odc.geobox is not None
    # TODO: More check for georefence

    # Check for necessary attributes
    assert "tile_id" in tile.attrs
    assert "azimuth" in tile.attrs
    assert "elevation" in tile.attrs

    for band in ["red", "green", "blue", "nir"]:
        assert "data_source" in tile[band].attrs, f"data_source not found in {band=}"
        assert "long_name" in tile[band].attrs, f"long_name not found in {band=}"
        assert "units" in tile[band].attrs, f"units not found in {band=}"

    print(tile["quality_data_mask"].attrs)

    assert "data_source" in tile["quality_data_mask"].attrs
    assert "long_name" in tile["quality_data_mask"].attrs
    assert "description" in tile["quality_data_mask"].attrs

    # Check if every band is in bound -0.1 to 7
    # Why 7? Because the max of uint16 is 65535 and 65535/10000 = 6.5535
    for band in ["red", "green", "blue", "nir"]:
        bmin = np.round(tile[band].min().item(), 4)  # Round to avoid float precision issues
        bmax = tile[band].max().item()
        assert bmin >= -0.1, f"{band} min value is {bmin}, but should be >=-0.1"
        assert bmax <= 7.0, f"{band} max value is {bmax}, but should be <= 7.0"

    # Check for stats and that the images are downloaded correctly
    if satellite == "s2":
        assert "s2_tile_id" in tile.attrs
        s2id = tile.attrs["s2_tile_id"]
        stats = S2_STATS.get(s2id, None)
        assert stats is not None, f"No stats found for {s2id=}"
    elif satellite == "planet":
        assert "planet_scene_id" in tile.attrs
        planetid = tile.attrs["tile_id"]
        stats = PLANET_STATS.get(planetid, None)
        assert stats is not None, f"No stats found for {planetid=}"
    else:
        raise ValueError(f"Unknown satellite {satellite}")
    for band in ["red", "green", "blue", "nir"]:
        if band in stats:
            expected_min = stats[band]["min"]
            expected_max = stats[band]["max"]
            expected_mean = stats[band]["mean"]
            expected_std = stats[band]["std"]
            np.testing.assert_allclose(
                tile[band].min(),
                expected_min,
                rtol=0.1,
                err_msg=f"{band} min value is {tile[band].min().values}, but should be {expected_min}",
            )
            np.testing.assert_allclose(
                tile[band].max(),
                expected_max,
                rtol=0.1,
                err_msg=f"{band} max value is {tile[band].max().values}, but should be {expected_max}",
            )
            np.testing.assert_allclose(
                tile[band].mean(),
                expected_mean,
                rtol=0.1,
                err_msg=f"{band} mean value is {tile[band].mean().values}, but should be {expected_mean}",
            )
            np.testing.assert_allclose(
                tile[band].std(),
                expected_std,
                rtol=0.1,
                err_msg=f"{band} std value is {tile[band].std().values}, but should be {expected_std}",
            )


# =============================================================================
# ===GEE=======================================================================
# =============================================================================


@pytest.mark.parametrize("s2id", GEE_IMAGES)
def test_load_gee_s2_sr_scene_download(s2id: str):
    ee_project = os.environ.get("EE_PROJECT")
    ee.Authenticate()
    ee.Initialize(project=ee_project)
    tile = load_gee_s2_sr_scene(s2id)
    assert_optical_dataset(tile, satellite="s2")


@pytest.mark.parametrize("s2id", GEE_IMAGES)
def test_load_gee_s2_sr_scene_from_cache(s2id: str):
    ee_project = os.environ.get("EE_PROJECT")
    ee.Authenticate()
    ee.Initialize(project=ee_project)
    tile = load_gee_s2_sr_scene(s2id, cache=DATA_DIR / "gee")
    assert_optical_dataset(tile, satellite="s2")


@pytest.mark.parametrize("s2id", GEE_IMAGES)
def test_load_gee_s2_sr_scene_caching(s2id: str):
    ee_project = os.environ.get("EE_PROJECT")
    ee.Authenticate()
    ee.Initialize(project=ee_project)
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Empty the cache directory
    for file in cache_dir.glob("*"):
        file.unlink()

    # First call downloads, second call uses cache
    downloaded_tile = load_gee_s2_sr_scene(s2id, cache=cache_dir)
    assert_optical_dataset(downloaded_tile, satellite="s2")
    cached_tile = load_gee_s2_sr_scene(s2id, cache=cache_dir)
    assert_optical_dataset(cached_tile, satellite="s2")
    assert downloaded_tile.identical(cached_tile)

    # Clean up cache directory
    for file in cache_dir.glob("*"):
        file.unlink()
    cache_dir.rmdir()


# =============================================================================
# ===STAC======================================================================
# =============================================================================


@pytest.mark.parametrize("s2id", STAC_IMAGES)
def test_load_cdse_s2_sr_scene_download(s2id: str):
    tile = load_cdse_s2_sr_scene(s2id)
    assert_optical_dataset(tile, satellite="s2")


@pytest.mark.parametrize("s2id", STAC_IMAGES)
def test_load_cdse_s2_sr_scene_from_cache(s2id: str):
    tile = load_cdse_s2_sr_scene(s2id, cache=DATA_DIR / "stac")
    assert_optical_dataset(tile, satellite="s2")


@pytest.mark.parametrize("s2id", STAC_IMAGES)
def test_load_cdse_s2_sr_scene_caching(s2id: str):
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Empty the cache directory
    for file in cache_dir.glob("*"):
        file.unlink()

    # First call downloads, second call uses cache
    downloaded_tile = load_cdse_s2_sr_scene(s2id, cache=cache_dir)
    assert_optical_dataset(downloaded_tile, satellite="s2")
    cached_tile = load_cdse_s2_sr_scene(s2id, cache=cache_dir)
    assert_optical_dataset(cached_tile, satellite="s2")
    assert downloaded_tile.identical(cached_tile)

    # Clean up cache directory
    for file in cache_dir.glob("*"):
        file.unlink()
    cache_dir.rmdir()


# =============================================================================
# ===PLANET====================================================================
# =============================================================================


@pytest.mark.parametrize("fpath", PLANET_SCENES)
def test_load_planet(fpath: str):
    tile = load_planet_scene(DATA_DIR / "planet" / fpath)
    mask = load_planet_masks(DATA_DIR / "planet" / fpath)
    tile = xr.merge([tile, mask])
    assert_optical_dataset(tile, satellite="planet")
