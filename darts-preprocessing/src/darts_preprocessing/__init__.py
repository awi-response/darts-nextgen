"""Data preprocessing and feature engineering for the DARTS dataset."""

import importlib.metadata

from darts_preprocessing.engineering.arcticdem import calculate_aspect as calculate_aspect
from darts_preprocessing.engineering.arcticdem import calculate_curvature as calculate_curvature
from darts_preprocessing.engineering.arcticdem import (
    calculate_dissection_index as calculate_dissection_index,
)
from darts_preprocessing.engineering.arcticdem import calculate_hillshade as calculate_hillshade
from darts_preprocessing.engineering.arcticdem import calculate_slope as calculate_slope
from darts_preprocessing.engineering.arcticdem import (
    calculate_terrain_ruggedness_index as calculate_terrain_ruggedness_index,
)
from darts_preprocessing.engineering.arcticdem import (
    calculate_topographic_position_index as calculate_topographic_position_index,
)
from darts_preprocessing.engineering.arcticdem import (
    calculate_vector_ruggedness_measure as calculate_vector_ruggedness_measure,
)
from darts_preprocessing.engineering.indices import (
    calculate_ctvi as calculate_ctvi,
)
from darts_preprocessing.engineering.indices import (
    calculate_evi as calculate_evi,
)
from darts_preprocessing.engineering.indices import (
    calculate_exg as calculate_exg,
)
from darts_preprocessing.engineering.indices import (
    calculate_gli as calculate_gli,
)
from darts_preprocessing.engineering.indices import (
    calculate_gndvi as calculate_gndvi,
)
from darts_preprocessing.engineering.indices import (
    calculate_grvi as calculate_grvi,
)
from darts_preprocessing.engineering.indices import calculate_ndvi as calculate_ndvi
from darts_preprocessing.engineering.indices import (
    calculate_nrvi as calculate_nrvi,
)
from darts_preprocessing.engineering.indices import (
    calculate_rvi as calculate_rvi,
)
from darts_preprocessing.engineering.indices import (
    calculate_savi as calculate_savi,
)
from darts_preprocessing.engineering.indices import (
    calculate_tgi as calculate_tgi,
)
from darts_preprocessing.engineering.indices import (
    calculate_ttvi as calculate_ttvi,
)
from darts_preprocessing.engineering.indices import (
    calculate_tvi as calculate_tvi,
)
from darts_preprocessing.engineering.indices import (
    calculate_vari as calculate_vari,
)
from darts_preprocessing.engineering.indices import (
    calculate_vdvi as calculate_vdvi,
)
from darts_preprocessing.engineering.indices import (
    calculate_vigreen as calculate_vigreen,
)
from darts_preprocessing.engineering.spyndex import calculate_spyndex as calculate_spyndex
from darts_preprocessing.legacy import preprocess_legacy_fast as preprocess_legacy_fast
from darts_preprocessing.v2 import preprocess_v2 as preprocess_v2

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
