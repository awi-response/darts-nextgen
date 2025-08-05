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
    Therefore, the memory representation of Sentinel 2 data is `float32` instead, but with the same range as the original data.

Therefore the data convertion happens in two different places of the pipeline:

- Disk -> Memory: Cache-Manager where data is loaded from a cache NetCDF file into memory.
  This should be handled via xarray, which follows the CF conventions.
- Memory -> Model: In the segmentation module, right before the data is transformed into PyTorch tensors for inference.
  For training, this is done before writing the data into the training dataset to save compute power and enable better caching.
- Model -> Memory: Never happens, the output propabilities are exported as is.
  Further, the probabilities are only appended to the data in memory-representation.
  Therefore the model-representation only ever exists in the inference or training code.
- Memory -> Disk: At the export module and with the Cache-Manager when writing cache files.
  This is done via xarray, which follows the CF conventions.

!!! note "Terminology"

    Because the data can have 3 different representations, it becomes unclear what is meant by "encoded" and "decoded".
    In general, the "Memory" representation is always the "true" and therefore "decoded" representation.
    However, outside of the context of convertions, we may use a following terminology:

    - **Encoded**: The data in the representation that is used for caching and exports, i.e. disk-representation.
    - **Decoded**: The data in the representation that is used for working and visualisation, i.e. memory-representation.
    - **Normalised**: The data in the representation that is used for training and inference, i.e. model-representation.

| DataVariable             | shape  | dtype (memory) | dtype(disk) | valid-range | disk-range | no-data (disk) | attrs                         | source                  | note |
| ------------------------ | ------ | -------------- | ----------- | ----------- | ---------- | -------------- | ----------------------------- | ----------------------- | ---- |
| `blue`                   | (x, y) | float32        | uint16      |             |            |                | data_source, long_name, units | PLANET / S2             |      |
| `green`                  | (x, y) | float32        | uint16      |             |            |                | data_source, long_name, units | PLANET / S2             |      |
| `red`                    | (x, y) | float32        | uint16      |             |            |                | data_source, long_name, units | PLANET / S2             |      |
| `nir`                    | (x, y) | float32        | uint16      |             |            |                | data_source, long_name, units | PLANET / S2             |      |
| `dem`                    | (x, y) | float32        |             | [0,[        |            |                | data_source, long_name, units | SmartGeocubes           |      |
| `dem_datamask`           | (x, y) | bool           | bool        | True/False  | True/False | False          | data_source, long_name, units | SmartGeocubes           |      |
| `tc_brightness`          | (x, y) | uint8          | uint8       | [0, 255]    | [0, 255]   | -              | data_source, long_name        | EarthEngine             |      |
| `tc_greenness`           | (x, y) | uint8          | uint8       | [0, 255]    | [0, 255]   | -              | data_source, long_name        | EarthEngine             |      |
| `tc_wetness`             | (x, y) | uint8          | uint8       | [0, 255]    | [0, 255]   | -              | data_source, long_name        | EarthEngine             |      |
| `ndvi`                   | (x, y) | float32        | uint16      | [-1, 1]     | [0, 20000] |                | data_source, long_name        | Preprocessing           |      |
| `relative_elevation`     | (x, y) | float32        |             |             |            |                | data_source, long_name, units | Preprocessing           |      |
| `slope`                  | (x, y) | float32        |             | [0, 90]     |            |                | data_source, long_name        | Preprocessing           |      |
| `aspect`                 | (x, y) | float32        |             | [0, 360]    |            |                | data_source, long_name        | Preprocessing           |      |
| `hillshade`              | (x, y) | float32        |             | [0, 1]      |            |                | data_source, long_name        | Preprocessing           |      |
| `curvature`              | (x, y) | float32        |             |             |            |                | data_source, long_name        | Preprocessing           |      |
| `probabilities`          | (x, y) | float32        |             | [0, 1]      |            |                | long_name                     | Ensemble / Segmentation |      |
| `probabilities-model-X*` | (x, y) | float32        |             | [0, 1]      |            |                | long_name                     | Ensemble / Segmentation |      |
| `probabilities_percent`  | (x, y) | uint8          |             | [0, 100]    |            | 255            | long_name, units              | Postprocessing          |      |
| `binarized_segmentation` | (x, y) |                |             | 0/1         |            | -              | long_name                     | Postprocessing          |      |

- `*` = Model name, e.g. `probabilities-model-UNet`, `probabilities-model-ResNet`, etc.
- The `no-data` value in memory for `float32` is always `nan`.

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

The scale and offset can be simply derived from the disk-range:

```py
offset = disk_range[0]
scale = disk_range[1] - disk_range[0]
```
