BandMapping = dict[str, list[str]]
NormalizationFactors = dict[str, int | dict[str, int]]

BAND_MAPPING: BandMapping = {
    # "optical": ["blue", "green", "red", "nir"],
    "planet": ["blue", "green", "red", "nir"],
    "s2": ["blue", "green", "red", "nir"],
    "s2-upscaled": ["blue", "green", "red", "nir"],
    "indices": ["ndvi"],
    "artic-dem": ["slope", "relative_elevation"],
    "tcvis": ["tc_brightness", "tc_greenness", "tc_wetness"],
}

NORMALIZATION_FACTORS = {
    "planet": 3000,
    "s2": 3000,
    "s2-upscaled": 3000,
    "indices": 1,
    "artic-dem": {
        "slope": 90,
        "relative_elevation": 30000,
    },
    "tcvis": 255,
}
