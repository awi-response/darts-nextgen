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
  - Training-Preprocessors
  - NOT in the export -> The export is handled manually
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

| DataVariable                | usage | shape  | dtype (memory) | dtype(disk) | valid-range   | disk-range    | no-data (disk) | attrs                               | source                  | note                                                                                      |
| --------------------------- | ----- | ------ | -------------- | ----------- | ------------- | ------------- | -------------- | ----------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------- |
| `blue`                      | inp   | (x, y) | float32        | uint16      | [-0.1, 0.5]   | [0, 65535]    | 0              | data_source, long_name, units       | PLANET / S2             |                                                                                           |
| `green`                     | inp   | (x, y) | float32        | uint16      | [-0.1, 0.5]   | [0, 65535]    | 0              | data_source, long_name, units       | PLANET / S2             |                                                                                           |
| `red`                       | inp   | (x, y) | float32        | uint16      | [-0.1, 0.5]   | [0, 65535]    | 0              | data_source, long_name, units       | PLANET / S2             |                                                                                           |
| `nir`                       | inp   | (x, y) | float32        | uint16      | [-0.1, 0.5]   | [0, 65535]    | 0              | data_source, long_name, units       | PLANET / S2             |                                                                                           |
| `s2_scl`                    | qal   | (x, y) | uint8          | uint8       | [0, 11]       | [0, 11]       | -              | data_source, long_name              | S2                      | <https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/> |
| `planet_udm`                | qal   | (x, y) | uint8          | uint8       | [0, 8]        | [0, 8]        | -              |                                     | PLANET                  | <https://docs.planet.com/data/imagery/udm/>                                               |
| `quality_data_mask`         | qal   | (x, y) | uint8          | uint8       | {0, 1, 2}     | {0, 1, 2}     | -              | data_source, long_name, description | Acquisition             | 0 = Invalid, 1 = Low Quality, 2 = High Quality                                            |
| `dem`                       | inp   | (x, y) | float32        | int16       | [-100, 3000]  | [0, 31000]    | -1             | data_source, long_name, units       | SmartGeocubes           |                                                                                           |
| `arcticdem_data_mask`       | qal   | (x, y) | uint8          | bool        | {0, 1}        | {False, True} | -              | data_source, long_name, units       | SmartGeocubes           |                                                                                           |
| `tc_brightness`             | inp   | (x, y) | uint8          | uint8       | [0, 255]      | [0, 255]      | -              | data_source, long_name              | EarthEngine             |                                                                                           |
| `tc_greenness`              | inp   | (x, y) | uint8          | uint8       | [0, 255]      | [0, 255]      | -              | data_source, long_name              | EarthEngine             |                                                                                           |
| `tc_wetness`                | inp   | (x, y) | uint8          | uint8       | [0, 255]      | [0, 255]      | -              | data_source, long_name              | EarthEngine             |                                                                                           |
| `ndvi`                      | inp   | (x, y) | float32        | int16       | [-1, 1]       | [0, 20000]    | -1             | long_name                           | Preprocessing           |                                                                                           |
| `relative_elevation`        | inp   | (x, y) | float32        | int16       | [-50, 50]     | [0, 30000]    | -1             | data_source, long_name, units       | Preprocessing           |                                                                                           |
| `slope`                     | inp   | (x, y) | float32        | int16       | [0, 90]       | [0, 9000]     | -1             | data_source, long_name              | Preprocessing           |                                                                                           |
| `aspect`                    | inp   | (x, y) | float32        | int16       | [0, 360]      | [0, 3600]     | -1             | data_source, long_name              | Preprocessing           |                                                                                           |
| `hillshade`                 | inp   | (x, y) | float32        | int16       | [0, 1]        | [0, 10000]    | -1             | data_source, long_name              | Preprocessing           |                                                                                           |
| `curvature`                 | inp   | (x, y) | float32        | int16       | [-1, 1]       | [0, 20000]    | -1             | data_source, long_name              | Preprocessing           |                                                                                           |
| `probabilities`             | dbg   | (x, y) | float32        | uint8       | [0, 1]        | [0, 100]      | 255            | long_name                           | Ensemble / Segmentation |                                                                                           |
| `probabilities-X*`          | dbg   | (x, y) | float32        | uint8       | [0, 1]        | [0, 100]      | 255            | long_name                           | Ensemble / Segmentation |                                                                                           |
| `binarized_segmentation`    | out   | (x, y) | bool           | bool        | {False, True} | {False, True} | -              | long_name                           | Postprocessing          |                                                                                           |
| `binarized_segmentation-X*` | dbg   | (x, y) | bool           | bool        | {False, True} | {False, True} | -              | long_name                           | Postprocessing          |                                                                                           |
| `extent`                    | out   | (x, y) | bool           | bool        | {False, True} | {False, True} | -              | long_name                           | Postprocessing          |                                                                                           |

Notes:

- `X*` = Model name, e.g. `probabilities-tcvis`, `probabilities-notcvis`, etc.
- The `no-data` value in memory for `float32` is always `nan`.
- `bool` types before postprocessing must be represented as uint8 in memory for easy reprojection etc.
- Modes of usage:
  - `inp`: (Potential) Input to the model
  - `qal`: Quality Assurance Layer, not used as input to the model, but for masking or filtering
  - `dbg`: Only exported for debugging purposes
  - `out`: Output of the model

!!! warning "Incompleteness"

    `attrs` is outdated.
    
    Missing:

    - New DEM Engineered - VRM DI etc.
    - New Indices - TGI, EXG, GLI etc.

!!! danger "Loss of Information"

    Because we encode almost every variable we work with into a smaller sized representation or into a smaller range, information get's lost.
    E.g. when writing the DEM to disk, values larger than 3000m will be clipped to 3000m and the minimum step size between values reduces to 0.1m.
    This is enough for our purposes, but may not be suitable for other applications.

### Optical bands: PLANET vs. S2-Harmonized (GEE) vs. Sentinel-2 L2A (Copernicus)

This is complicated: _in theory_ the range of this data is between 0 and 1 and measured as surface reflectance.
However, the values can be negative (e.g. due to atmospheric correction) and larger than 1 (e.g. due to bright surfaces like snow).
Thus, Copernicus applies a shift of -0.1 to allow for negative values in the Sentinel-2 L2A product.
This was introduced in 2022 - Google Earth Engine just reverts the shift for all data after 2022 in the S2-Harmonized collection, because they never re-upload the data.
Once uploaded, the data is fixed and will never change.
Thus, GEE S2-Harmonized data is lossy.
The unharmonized dataset in GEE, which is officially deprecated, is not lossy, but spectral values are not comparable between years before and after 2022.
Further, because data is never re-uploaded, the processing of older imagery is different than newer imagery.
The data in GEE and in Copernicus is always stored as uint16 values between 0 and 65535 (maximum of uint16) with a scale factor of 10000, just with different offsets.

In our pipeline we want to be able to utilize negative values in the model.
Because most viable (non-snow, non-cloud) values are not larger than 0.5, we decided to use the range [-0.1, 0.5] for the memory representation.
Of course, this only applies to the normalization before the data is fed into the model, thus calculating indices like NDVI are not limited to this range.

Data which is directly downloaded from either Copernicus or GEE is directly stored in the cache with their own representation.
This is not documented in the table above and is specific to the acquisition module.
All data which is output from the acquisition module is always converted to the memory representation.

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
For that, the attributes `_FillValue`, `scale_factor`, and `add_offset` are set by a helper module `darts_utils.bands.BandManager`.
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

### Legacy support

In order to support legacy models, it is necessary to check which model version was used.
For this, from now on all checkpoints get a new field `model_version` in their metadata.
Fortunatly, all previous normalizations are equal to the new ones, hence to remapping needs to be done.
