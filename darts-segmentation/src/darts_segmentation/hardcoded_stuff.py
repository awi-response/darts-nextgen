BAND_MAPPING = {
    "planet": ["red", "green", "blue", "nir"],
    "s2": ["red", "green", "blue", "nir"],
    "s2-upscaled": ["red", "green", "blue", "nir"],
    "ndvi": ["ndvi"],
    "slope": ["slope"],
    "relative_elevation": ["relative_elevation"],
    "tcvis": ["tc_brightness", "tc_greenness", "tc_wetness"],
}

NORMALIZATION_FACTORS = {
    "planet": 3000,
    "s2": 3000,
    "s2-upscaled": 3000,
    "ndvi": 1,
    "slope": 90,
    "relative_elevation": 30000,
    "tcvis": 255,
}
