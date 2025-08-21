"""Hardcoded band information for encoding/decoding and normalization."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


def _get_dtype_min_max(dtype: str) -> tuple[int | float, int | float]:
    if dtype.startswith("int") or dtype.startswith("uint"):
        return np.iinfo(dtype).min, np.iinfo(dtype).max
    elif dtype.startswith("float"):
        return np.finfo(dtype).min, np.finfo(dtype).max
    elif dtype == "bool":
        return 0, 1
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


_supported_dtypes = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]


@dataclass
class BandCodec:
    """Encoding / Decoding information for a single band (channel).

    Stores information about how to convert data between the three different representations:

    - memory: the in-memory representation of the data (native)
    - disk: the on-disk representation of the data (best compression)
    - model: the representation used for model training & inference (normalized between 0 and 1)

    In general the "default" representation is the memory representation and is further referred to as "decoded".
    The disk and model representations are referred to as "encoded", both with their own scale factors and offsets.
    The scale and offset of the disk representation must be chosen such that the encoded values fit into the disk dtype.
    The model representation is normalized to the range [0, 1] using the valid range of the memory representation.
    The formulas used are based on the NetCDF conventions for encoding and decoding data used by Xarray:
    https://docs.xarray.dev/en/stable/user-guide/io.html#scaling-and-type-conversions

    ```py
    decoded = scale_factor * encoded + add_offset
    encoded = (decoded - add_offset) / scale_factor
    ```

    """

    disk_dtype: str
    memory_dtype: str
    valid_range: tuple[float | int, float | int]
    scale_factor: float | None = None
    offset: float | None = None
    fill_value: float | int | None = None

    @property
    def disk_range(self) -> tuple[float | int, float | int]:
        """Range of the disk representation."""
        return (
            (self.valid_range[0] - (self.offset or 0)) / (self.scale_factor or 1),
            (self.valid_range[1] - (self.offset or 0)) / (self.scale_factor or 1),
        )

    @property
    def norm_factor(self) -> float:
        """Normalization factor for the model representation."""
        if self.memory_dtype == "bool":
            return 1.0
        return self.valid_range[1] - self.valid_range[0]

    @property
    def norm_offset(self) -> float:
        """Normalization offset for the model representation."""
        if self.memory_dtype == "bool":
            return 0.0
        return self.valid_range[0]

    @classmethod
    def bool(cls) -> "BandCodec":
        """Create a BandCodec for boolean bands.

        Boolean bands are represented as `bool` in memory and on disk, with a valid range of (False, True).
        They do not have a scale factor or offset, and the fill value is always None.

        Returns:
            BandCodec: A BandCodec instance for boolean bands.

        """
        return cls(
            disk_dtype="bool",
            memory_dtype="bool",
            valid_range=(False, True),
        )

    @classmethod
    def percentage(cls) -> "BandCodec":
        """Create a BandCodec for percentage bands.

        Percentage bands are represented as `float32` in memory and as `uint8` on disk,
        with a valid range of (0.0, 1.0) in memory and (0, 100) on disk with 255 as NoData.

        Returns:
            BandCodec: A BandCodec instance for percentage bands.

        """
        return cls(
            disk_dtype="uint8",
            memory_dtype="float32",
            valid_range=(0.0, 1.0),
            scale_factor=1 / 100,
            offset=0.0,
            fill_value=255,
        )

    @classmethod
    def ndi(cls) -> "BandCodec":
        """Create a BandCodec for Normalized Difference Index (NDI) bands.

        NDI bands are represented as `float32` in memory and as `int16` on disk,
        with a valid range of (-1.0, 1.0) in memory and (0, 20000) on disk with -1 as NoData.

        Returns:
            BandCodec: A BandCodec instance for NDI bands.

        """
        return cls(
            disk_dtype="int16",
            memory_dtype="float32",
            valid_range=(-1.0, 1.0),
            scale_factor=1 / 10000,
            offset=-1.0,
            fill_value=-1,
        )

    @classmethod
    def tc(cls) -> "BandCodec":
        """Create a BandCodec for tcvis data.

        TCVis bands are represented as `uint8` in memory and on dask, utilizing the complete 0-255 range.
        There are no NoData values.

        Returns:
            BandCodec: A BandCodec instance for TCVis bands.

        """
        return cls(
            disk_dtype="uint8",
            memory_dtype="uint8",
            valid_range=(0, 255),
        )

    @classmethod
    def optical(cls) -> "BandCodec":
        """Create a BandCodec for optical satellite imagery.

        Optical imagery bands are represented as `float32` in memory and as `uint16` on disk,
        with a valid range of (0, 10000) in memory and on disk with 0 as NoData.

        Please see the documentation about bands for caveats with optical data.

        Returns:
            BandCodec: A BandCodec instance for optical bands.

        """
        return cls(
            disk_dtype="uint16",
            memory_dtype="float32",
            valid_range=(0, 10000),
            fill_value=0,
        )

    @classmethod
    def mask(cls, vmax: int) -> "BandCodec":
        """Create a BandCodec for non-binary masks.

        Assumes the mask always start with 0

        Args:
            vmax (int): Maximum value of the mask

        Returns:
            BandCodec: A BandCodec instance for non-binary masks.

        """
        return cls(
            disk_dtype="uint8",
            memory_dtype="uint8",
            valid_range=(0, vmax),
        )

    def validate(self) -> str | None:  # noqa: C901
        """Validate the codec configuration.

        Checks if the disk representation's valid range fits within the limits of the disk dtype,
        and if the model representation's valid range is normalized between 0 and 1.

        Further checks the validity of the data types for scale factor, offset, and fill value.

        Returns:
            str | None: Reason for invalid configuration if any, otherwise None.

        """
        # ?: This complete function is written to be easily readable and understandable.
        # Of course, lot of if statements could be chained / combined, but that would make it harder to read.

        # Check dtype compatibility
        if self.memory_dtype not in _supported_dtypes:
            return f"Unsupported memory dtype: {self.memory_dtype}"

        if self.disk_dtype not in _supported_dtypes:
            return f"Unsupported disk dtype: {self.disk_dtype}"

        is_bool = self.memory_dtype == "bool"
        is_float = self.memory_dtype.startswith("float")

        # Check range validity
        if not is_bool:
            disk_range = (
                (self.valid_range[0] - (self.offset or 0)) / (self.scale_factor or 1),
                (self.valid_range[1] - (self.offset or 0)) / (self.scale_factor or 1),
            )
            disk_dtype_min, disk_dtype_max = _get_dtype_min_max(self.disk_dtype)
            if disk_range[0] < disk_dtype_min or disk_range[1] > disk_dtype_max:
                return (
                    f"Disk range {disk_range} is out of bounds "
                    f"for dtype {self.disk_dtype} ({disk_dtype_min}, {disk_dtype_max})"
                )

            norm_range = (
                (self.valid_range[0] - self.norm_offset) / self.norm_factor,
                (self.valid_range[1] - self.norm_offset) / self.norm_factor,
            )
            if norm_range[0] < 0 or norm_range[1] > 1:
                return (
                    f"Model range {norm_range} is out of bounds for normalized representation "
                    "(should be between 0 and 1)"
                )
        else:
            # For boolean bands, valid range is always (False, True)
            if self.valid_range != (False, True):
                return "Boolean bands must have valid range (False, True)"

        # Check scale factor and offset validity
        if self.scale_factor is not None or self.offset is not None:
            if not is_float:
                return "Integer and Boolean bands must not have scale factor or offset"
            # Check if one is None
            if self.scale_factor is None or self.offset is None:
                return "Float bands must have both or none of scale factor and offset defined"

        # Check fill value validity
        if is_float:
            if not isinstance(self.fill_value, float | int) and self.fill_value is not None:
                return "Float bands must have float or integer fill value if present"
        else:
            if self.fill_value is not None:
                return "Integer and Boolean bands must not have fill value"

        return None


@dataclass
class BandManager:
    """Meta class for loading, storing and encoding xarray datasets based on band codecs.

    Supports wildcard patterns for band names, e.g. "probabilities_*"
    to match all dataset variables starting with "probabilities_".
    """

    codecs: dict[str, BandCodec]

    def __getitem__(self, selector: list[str] | str) -> dict[str, BandCodec] | BandCodec:
        """Get a subset of codecs by band names or a single codec by band name.

        Args:
            selector (list[str] | str): A list of band names or a single band name to select.

        Returns:
            dict[str, BandCodec]: A dictionary of selected codecs.

        Raises:
            KeyError: If the band name is not found in the codecs.

        """
        if isinstance(selector, str):
            codec = self.codecs.get(selector)
            if codec is None:
                wildcard_bands = [band for band in self.codecs if "*" in band]
                codec = next((self.codecs[wb] for wb in wildcard_bands if wb.replace("*", "") in selector), None)
                if codec is None:
                    raise KeyError(f"Band '{selector}' not found in codecs.")
            return codec

        return BandManager({band: self.codecs[band] for band in selector if band in self.codecs})

    def __contains__(self, band: str) -> bool:
        """Check if a band is present in the manager.

        Args:
            band (str): The band name to check.

        Returns:
            bool: True if the band is present, False otherwise.

        """
        return band in self.codecs

    def __iter__(self):
        """Iterate over the bands in the manager.

        Yields:
            str: The band names in the manager.

        """
        yield from self.codecs

    def get(self, selector: str) -> BandCodec | None:
        """Get a codec by band name.

        Args:
            selector (str): The band name to select.

        Returns:
            BandCodec | None: The codec for the band, or None if not found.

        """
        try:
            return self[selector]
        except KeyError:
            return None

    def validate(self):
        """Validate all codecs in the manager.

        Iterates through all codecs and checks if they are valid.

        Raises:
            ValueError: If any codec is invalid, with a message indicating the band and reason for

        """
        for band, codec in self.codecs.items():
            invalid_reason = codec.validate()
            if invalid_reason:
                raise ValueError(f"Validation failed for {band=}: {invalid_reason}")

    def normalize(self, dataset: xr.Dataset) -> xr.Dataset:
        """Normalize the dataset to the model representation.

        Applies the normalization formula to each band in the dataset based on the codec configuration.

        Leaves boolean bands as they are, just converting them to float32.
        Also fills NaN values with 0.0 after normalization.
        All other bands are normalized to the range [0, 1] using the valid range of the memory representation
        and converted to float32.

        Args:
            dataset (xr.Dataset): The dataset to normalize.

        Returns:
            xr.Dataset: The normalized dataset.

        Raises:
            ValueError: If the dataset has unknown bands / channels.

        """
        # ?: We do not provide a default codec, so we cannot normalize if the dataset has unknown bands.
        # This is a design choice, because we want to ensure that we can track down what went into the model.
        # Note: Wildcard bands are not supported here for the same reason.
        if not set(dataset).issubset(self):
            raise ValueError(f"Dataset has unknown bands: {set(dataset) - set(self)}")
        dataset = dataset.copy(deep=True)
        for band in dataset:
            if dataset[band].dtype == "bool":
                # Convert boolean to float to it is already 0-1
                dataset[band] = dataset[band].astype("float32")
                continue
            codec = self.codecs[band]  # Safe because we checked above
            dataset[band] = (
                ((dataset[band] - codec.norm_offset) / codec.norm_factor).astype("float32").fillna(0.0).clip(0.0, 1.0)
            )
        return dataset

    def _get_encodings(self, dataset: xr.Dataset) -> dict[str, dict[str, str | float | int]]:
        # Create encoding information based on the codecs
        encodings = {}
        for band in dataset:
            codec = self.get(band)
            if codec is None:
                continue  # Skip bands not in codecs
            assert codec.memory_dtype == dataset[band].dtype, (
                f"Memory dtype mismatch for {band}: expected {codec.memory_dtype} but got {dataset[band].dtype}"
            )
            # Only float memory dtypes can be encoded
            if codec.memory_dtype != "float32" and codec.memory_dtype != "float64":
                continue
            encodings[band] = {"dtype": codec.disk_dtype}
            if codec.fill_value is not None:
                encodings[band]["_FillValue"] = codec.fill_value
            if codec.scale_factor is not None:
                encodings[band]["scale_factor"] = codec.scale_factor
            if codec.offset is not None:
                encodings[band]["add_offset"] = codec.offset
        return encodings

    def crop(self, dataset: xr.Dataset) -> xr.Dataset:
        """Crop the dataset to the valid range of each band.

        Clips each band in the dataset to its valid range defined in the codec.
        This is useful for ensuring that the data fits into the encoding.

        !!! warning "Inplace operation"

            This operation happens inplace - hence the data of the input dataset is changed.

        Args:
            dataset (xr.Dataset): The dataset to crop.

        Returns:
            xr.Dataset: The cropped dataset.

        """
        for band in dataset:
            codec = self.get(band)
            if codec is None:
                continue
            min_val, max_val = codec.valid_range
            dataset[band] = dataset[band].clip(min=min_val, max=max_val)
        return dataset

    def to_netcdf(self, dataset: xr.Dataset, path: Path | str, crop: bool = True) -> None:
        """Store the dataset to a NetCDF file.

        Args:
            dataset (xr.Dataset): The dataset to store.
            path (Path | str): The path to the NetCDF file.
            crop (bool): Whether to crop the dataset to the valid range. This happens inplace! Defaults to True.

        """
        path = Path(path)
        encodings = self._get_encodings(dataset)
        if crop:
            dataset = self.crop(dataset)
        dataset.to_netcdf(
            path,
            encoding=encodings,
            engine="h5netcdf",
        )

    def open(self, path: Path | str) -> xr.Dataset:
        """Load a dataset from a NetCDF file.

        Args:
            path (Path | str): The path to the NetCDF file.

        Returns:
            xr.Dataset: The loaded dataset.

        """
        dataset = xr.open_dataset(path, engine="h5netcdf", decode_coords="all", decode_cf=True).load()
        # Change the dtypes to the memory representation
        for band in dataset:
            codec = self.get(band)
            if codec is None:
                continue
            if dataset[band].dtype != codec.memory_dtype:
                dataset[band] = dataset[band].astype(codec.memory_dtype)

        return dataset


# Singleton instance of BandManager with predefined codecs
manager = BandManager(
    {
        "blue": BandCodec.optical(),
        "red": BandCodec.optical(),
        "green": BandCodec.optical(),
        "nir": BandCodec.optical(),
        "B02_10m": BandCodec.optical(),
        "B03_10m": BandCodec.optical(),
        "B04_10m": BandCodec.optical(),
        "B08_10m": BandCodec.optical(),
        "SCL_20m": BandCodec.mask(11),
        "s2_scl": BandCodec.mask(11),
        "planet_udm": BandCodec.mask(8),
        "quality_data_mask": BandCodec.mask(2),
        "dem": BandCodec(
            disk_dtype="float32",
            memory_dtype="float32",
            valid_range=(-100, 3000),
            scale_factor=0.1,
            offset=-100.0,
            fill_value=-1,
        ),
        "arcticdem_data_mask": BandCodec(
            disk_dtype="bool",
            memory_dtype="uint8",
            valid_range=(0, 1),
        ),
        "tc_brightness": BandCodec.tc(),
        "tc_greenness": BandCodec.tc(),
        "tc_wetness": BandCodec.tc(),
        "ndvi": BandCodec.ndi(),
        "relative_elevation": BandCodec(
            disk_dtype="int16",
            memory_dtype="float32",
            valid_range=(-50, 50),
            scale_factor=100 / 30000,
            offset=-50.0,
            fill_value=-1,
        ),
        "slope": BandCodec(
            disk_dtype="int16",
            memory_dtype="float32",
            valid_range=(0, 90),
            scale_factor=1 / 100,
            offset=0.0,
            fill_value=-1,
        ),
        "aspect": BandCodec(
            disk_dtype="int16",
            memory_dtype="float32",
            valid_range=(0, 360),
            scale_factor=1 / 10,
            offset=0.0,
            fill_value=-1,
        ),
        "hillshade": BandCodec(
            disk_dtype="int16",
            memory_dtype="float32",
            valid_range=(0, 1),
            scale_factor=1 / 10000,
            offset=0.0,
            fill_value=-1,
        ),
        "curvature": BandCodec.ndi(),  # Has the same properties (valid range is -1, 1) as NDIs
        "probabilities": BandCodec.percentage(),
        "probabilities-*": BandCodec.percentage(),
        "binarized_segmentation": BandCodec.bool(),
        "binarized_segmentation-*": BandCodec.bool(),
        "extent": BandCodec.bool(),
    }
)
manager.validate()
