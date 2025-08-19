# Band / Modalities and Normalisation

In the training dataset preparation, all bands available from the preprocessing will be included into the train dataset.
They are normalised and clipped to the range [0, 1], by hard-coded normalisation factors and offsets.
When training, a subset of available bands can be selected, on which the model should be trained.
This enables the possibility to quickly test different band combinations and their influence on the model performance without the need to preprocess the data again.

The information about which bands used for training is the written into the model checkpoint.
This information is then used to select the bands for inference.

## Representations states

Split up the data representation into three different representations:

- **Disk**: Data is stored in the most efficient way, e.g. uint16 for Sentinel 2, uint8 for TCVIS
- **Memory**: Data is stored in the most convenient and correct way for visualisation purposes. This should be equal to the original data representation. E.g. [-1, 1] float for NDVI
- **Model**: Data ist normalised to [0, 1] for training and inference and is always `float32`

!!! note "Memory representation exeptions"

    The memory representation is not always the same as the original data representation.
    For example, Satellite data like Sentinel 2 is originally stored as `uint16`, however, we also want to account for NaN values in the data.
    Therefore, the memory representation of Sentinel 2 data is `float32` instead, but with the same range as the original data and 0 replaced with NaN.

### Specification Details

The data convertion happens in three different places of the pipeline:

- _Disk -> Memory_:
  - Cache-Manager where data is loaded from a cache NetCDF file into memory.
    This should be handled via xarray, which follows the CF conventions.
  - Export functions
- _Memory -> Model_: In the segmentation module, right before the data is transformed into PyTorch tensors for inference.
  For training, this is done before writing the data into the training dataset to save compute power and enable better caching.
- _Model -> Memory_: Never happens, the output propabilities are exported as is.
  Further, the probabilities are only appended to the data in memory-representation.
  Therefore the model-representation only ever exists in the inference or training code.
- _Memory -> Disk_: At the export module and with the Cache-Manager when writing cache files.
  This is done via xarray, which follows the CF conventions.

!!! note "Terminology"

    Because the data can have 3 different representations, it becomes unclear what is meant by "encoded" and "decoded".
    In general, the "Memory" representation is always the "true" and therefore "decoded" representation.
    However, outside of the context of convertions, we may use a following terminology:

    - **Encoded**: The data in the representation that is used for caching and exports, i.e. disk-representation.
    - **Decoded**: The data in the representation that is used for working and visualisation, i.e. memory-representation.
    - **Normalised**: The data in the representation that is used for training and inference, i.e. model-representation.

| DataVariable             | shape  | dtype (memory) | dtype(disk) | valid-range  | disk-range | no-data (disk) | attrs                               | source                  | note                                           |
| ------------------------ | ------ | -------------- | ----------- | ------------ | ---------- | -------------- | ----------------------------------- | ----------------------- | ---------------------------------------------- |
| `blue`                   | (x, y) | float32        | uint16      | [0, 10000]   | [0, 65535] | 0              | data_source, long_name, units       | PLANET / S2             |                                                |
| `green`                  | (x, y) | float32        | uint16      | [0, 10000]   | [0, 65535] | 0              | data_source, long_name, units       | PLANET / S2             |                                                |
| `red`                    | (x, y) | float32        | uint16      | [0, 10000]   | [0, 65535] | 0              | data_source, long_name, units       | PLANET / S2             |                                                |
| `nir`                    | (x, y) | float32        | uint16      | [0, 10000]   | [0, 65535] | 0              | data_source, long_name, units       | PLANET / S2             |                                                |
| `quality_data_mask`      | (x, y) | uint8          | uint8       | {0, 1, 2}    | {0, 1, 2}  | -              | data_source, long_name, description | PLANET / S2             | 0 = Invalid, 1 = Low Quality, 2 = High Quality |
| `dem`                    | (x, y) | float32        | int16       | [-100, 3000] | [0, 31000] | -1             | data_source, long_name, units       | SmartGeocubes           |                                                |
| `dem_datamask`           | (x, y) | uint8          | bool        | 0/1          | False/True | -              | data_source, long_name, units       | SmartGeocubes           |                                                |
| `tc_brightness`          | (x, y) | uint8          | uint8       | [0, 255]     | [0, 255]   | -              | data_source, long_name              | EarthEngine             |                                                |
| `tc_greenness`           | (x, y) | uint8          | uint8       | [0, 255]     | [0, 255]   | -              | data_source, long_name              | EarthEngine             |                                                |
| `tc_wetness`             | (x, y) | uint8          | uint8       | [0, 255]     | [0, 255]   | -              | data_source, long_name              | EarthEngine             |                                                |
| `ndvi`                   | (x, y) | float32        | int16       | [-1, 1]      | [0, 20000] | -1             | data_source, long_name              | Preprocessing           |                                                |
| `relative_elevation`     | (x, y) | float32        | int16       | [-50, 50]    | [0, 30000] | -1             | data_source, long_name, units       | Preprocessing           |                                                |
| `slope`                  | (x, y) | float32        | int16       | [0, 90]      | [0, 9000]  | -1             | data_source, long_name              | Preprocessing           |                                                |
| `aspect`                 | (x, y) | float32        | int16       | [0, 360]     | [0, 3600]  | -1             | data_source, long_name              | Preprocessing           |                                                |
| `hillshade`              | (x, y) | float32        | int16       | [0, 1]       | [0, 10000] | -1             | data_source, long_name              | Preprocessing           |                                                |
| `curvature`              | (x, y) | float32        | int16       | [-1, 1]      | [0, 20000] | -1             | data_source, long_name              | Preprocessing           |                                                |
| `probabilities`          | (x, y) | float32        | uint8       | [0, 1]       | [0, 100]   | 255            | long_name                           | Ensemble / Segmentation |                                                |
| `probabilities-model-X*` | (x, y) | float32        | uint8       | [0, 1]       | [0, 100]   | 255            | long_name                           | Ensemble / Segmentation |                                                |
| `binarized_segmentation` | (x, y) | bool           | bool        | False/True   | False/True | -              | long_name                           | Postprocessing          |                                                |
| `extent`                 | (x, y) | bool           | bool        | False/True   | False/True | -              | long_name                           | Postprocessing          |                                                |

- `*` = Model name, e.g. `probabilities-model-UNet`, `probabilities-model-ResNet`, etc.
- The `no-data` value in memory for `float32` is always `nan`.
- All `bool` disk-encoded values are of course True / False without nans (they are always equal to False).
- `bool` types before postprocessing must be represented as uint8 in memory for easy reprojection etc.
- Missing: New DEM Engineered: VRM DI etc.

!!! danger "Loss of Information"

    Because we encode almost every variable we work with into a smaller sized representation or into a smaller range, information get's lost.
    E.g. when writing the DEM to disk, values larger than 3000m will be clipped to 3000m and the minimum step size between values reduces to 0.1m.
    This is enough for our purposes, but may not be suitable for other applications.

### Optical bands: PLANET vs. Sentinel 2 (GEE) vs. Sentinel 2 (Copernicus)

This is complicated: _in theory_ the range of this data is between 0 and 65535 (maximum of uint16), since surface reflectance does not have a defined upper limit.
However, the satellite sensor has one: e.g. PLANET sensor outputs 12-bit integers, therefore a value of 4095 (2^12 - 1) is the maximum sensor value.
This value however, is further reprocessed and scaled by an unknown factor (this factor always depends on the specific image metadata).
Further, PLANET and Sentinel-2 are not color balanced to each other.
In case of Sentinel-2 L2A the postprocessing shifts values by 1000 to allow encoding of negative reflectance.
This was introduced in 2022 - Google Earth Engine just reverts the shift for future processings.
Hence, GEE and CDSE values are shifted by 1000.

The storage handling and normalization handling happens with a simplified approach:

- Even if the _theoretical_ valid data range of the optical data it is assumed that the valid data range is between 0 and 10000.
  Hence, values above will be clipped when normalizing, but only then.
  Therefore, the decoded representation in memory can be larger than 10000 and has dtype float.
- Normalization happens to be a linear scaling to the range [0, 1] based on the 10000 maximum value.
  All further quantization etc. happens after normalization.
- For visualization purposes, it is recommended to crop values to 3000.

### DEM

The highest point in the arctic is approx. 3000m.
The lowest depends on the geoid used, for arcticdem there are very few values below -10. (i guess)
Hence, the valid-range scaling is similar to the optical data arbitrary.

For TPI (relative_elevation), the valid-range strongly depends on the kernel used.
The range increases with larger kernel sizes.
E.g. some tests with Sentinel 2:

- 2px (20m) kernel: [-3, 3]
- 10px (100m) kernel: [-40, 20]
- 100px (1000m) kernel: [-60, 40]

Since we use mostly a kernel between 10px and 100px, we can expect the valid range to be between [-50, 50].

## Implementation Details

All Disk <-> Memory convertion are be done via [xarray through their CF convention layer](https://docs.xarray.dev/en/stable/user-guide/io.html#reading-encoded-data) (`decode_cf=True`)
For that, the attributes `_FillValue`, `scale_factor`, and `add_offset` are set by a helper module `darts_utils.bands.BandLoader`.
This helper is used by the Cache-Manager and the export module to ensure that the data is always in the correct representation.

!!! danger "_FillValue"

    With rioxarray it is possible to assign a `_FillValue` attribute to the data variables with `.rio.write_nodata()`.
    This can lead to weird behaviour when writing and reading the data:

    ```py
    >>> "Before writing with _FillValue=0.0: dtype=uint16, attrs={'_FillValue': 0.0, 'data_source': 'planet', 'long_name': 'NDVI'}"
    >>> "After reading with _FillValue=0.0: dtype=float32, attrs={'data_source': 'planet', 'long_name': 'NDVI'}"
    >>> "Before writing with _FillValue=0: dtype=uint16, attrs={'_FillValue': 0, 'data_source': 'planet', 'long_name': 'NDVI'}"
    >>> "After reading with _FillValue=0: dtype=float32, attrs={'data_source': 'planet', 'long_name': 'NDVI'}"
    >>> "Before writing wihtout _FillValue: dtype=uint16, attrs={'data_source': 'planet', 'long_name': 'NDVI'}"
    >>> "After reading without _FillValue: dtype=uint16, attrs={'data_source': 'planet', 'long_name': 'NDVI'}"
    ```

### Scale and Offset

The scale and offset for normalization is automatically derived from the `valid-range` parameter of a BandCodec.

For disk encoding the following formula can be used to derive the scale and offset manually based on the `valid-range` and the `disk-range`:

```py
offset = valid_range.min
scale = (valid_range.max - valid_range.min) / (disk_range.max - disk_range.min)
```

E.g. for NDVI with a `valid_range=(-1.0, 1.0)` and `disk_range=(0, 20000)`:

```py
> offset = valid_range.min
> scale = (valid_range.max - valid_range.min) / (disk_range.max - disk_range.min)
> offset, scale
-1., (2 / 20000) -> -1., 0.0001
```
