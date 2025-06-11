# Band / Modalities and Normalisation

In the training dataset preparation, all bands available from the preprocessing will be included into the train dataset.
They are normalised and clipped to the range [0, 1], by hard-coded normalisation factors and offsets.
When training, a subset of available bands can be selected, on which the model should be trained.
The information about which bands used for training is the written into the model checkpoint.
This information is then used to select the bands for inference.

## Current State

- Values & ranges after preprocessing
- Training Dataset preprocessing normalisation
- What is stored in the model
- How that is applied during inference

## Future

Split up the data representation into three different representations:

- **Disk**: Data is stored in the most efficient way, e.g. uint16 for Sentinel 2, uint8 for TCVIS
- **Memory**: Data is stored in the most convenient and correct way for visualisation purposes. E.g. [-1, 1] float for NDVI
- **Model**: Data ist normalised to [0, 1] for training and inference

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

    Because the data can have 3 different representations, instead of the usual two states (encoded and decoded), the terminology also changes a little.
    When talking about one of the convertions Disk -> Memory -> Model, the following applies to the traditional terminology:

    - **Encoded**: The data on the left side of the arrow.
    - **Decoded**: The data on the right side of the arrow.
    
    So, in the case of a convertion from disk to memory, the encoded data is the data in disk-representation and the decoded data is the data in memory-representation.
    In case of the convertion from memory to model, the encoded data is the data in memory-representation and the decoded data is the data in model-representation.

    However, outside of the context of convertions, we may use a different terminology:

    - **Encoded**: The data in the representation that is used for caching and exports, i.e. disk-representation.
    - **Decoded**: The data in the representation that is used for working and visualisation, i.e. memory-representation.
    - **Normalised**: The data in the representation that is used for training and inference, i.e. model-representation.

### Implementation plan

All Disk <-> Memory convertion should be done via [xarray through their CF convention layer](https://docs.xarray.dev/en/stable/user-guide/io.html#reading-encoded-data) (`decode_cf=True`)
For that, the attributes `_FillValue`, `scale_factor`, and `add_offset` should be set by the module which creates that data.
This also includes the Cache-Manager, even if it just reads the data from disk, because these attributes getting lost at read.

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

### Disk representation

!!! info

    This only applies to caching and exports of the data, not to the sources of the data.

| DataVariable             | shape  | dtype   | valid-range | no-data | attrs                         | source                  | note |
| ------------------------ | ------ | ------- | ----------- | ------- | ----------------------------- | ----------------------- | ---- |
| `blue`                   | (x, y) | uint16  | [0,3000+]   | 0       | data_source, long_name, units | PLANET / S2             |      |
| `green`                  | (x, y) | uint16  | [0,3000+]   | 0       | data_source, long_name, units | PLANET / S2             |      |
| `red`                    | (x, y) | uint16  | [0,3000+]   | 0       | data_source, long_name, units | PLANET / S2             |      |
| `nir`                    | (x, y) | uint16  | [0,3000+]   | 0       | data_source, long_name, units | PLANET / S2             |      |
| `dem`                    | (x, y) | float32 | [0,[        | nan     | data_source, long_name, units | SmartGeocubes           |      |
| `dem_datamask`           | (x, y) | bool    | True/False  | False   | data_source, long_name, units | SmartGeocubes           |      |
| `tc_brightness`          | (x, y) | uint8   | [0, 255]    | -       | data_source, long_name        | EarthEngine             |      |
| `tc_greenness`           | (x, y) | uint8   | [0, 255]    | -       | data_source, long_name        | EarthEngine             |      |
| `tc_wetness`             | (x, y) | uint8   | [0, 255]    | -       | data_source, long_name        | EarthEngine             |      |
| `ndvi`                   | (x, y) | uint16  | [0, 20000]  | 0       | data_source, long_name        | Preprocessing           |      |
| `relative_elevation`     | (x, y) | int16   |             | 0       | data_source, long_name, units | Preprocessing           |      |
| `slope`                  | (x, y) | float32 | [0, 90]     | nan     | data_source, long_name        | Preprocessing           |      |
| `aspect`                 | (x, y) | float32 | [0, 360]    | nan     | data_source, long_name        | Preprocessing           |      |
| `hillshade`              | (x, y) | float32 | [0, 1]      | nan     | data_source, long_name        | Preprocessing           |      |
| `curvature`              | (x, y) | float32 |             | nan     | data_source, long_name        | Preprocessing           |      |
| `probabilities`          | (x, y) | float32 |             | nan     | long_name                     | Ensemble / Segmentation |      |
| `probabilities-model-X*` | (x, y) | float32 |             | nan     | long_name                     | Ensemble / Segmentation |      |
| `probabilities_percent`  | (x, y) | uint8   | [0, 100]    | 255     | long_name, units              | Postprocessing          |      |
| `binarized_segmentation` | (x, y) | uint8   | 0/1         | -       | long_name                     | Postprocessing          |      |

TODO: Speak with Jonas and Ingmar about how we want to store things.

### Memory representation

The following scale and offset values are used for the normalisation:

TODO

### Model representation

The model representation of each band is normalised to the range `[0, 1]` and uses `float32`.
In addition to the convertion, the values are clipped to the range `[0, 1]` to avoid outliers.
The following scale and offset values are used for the normalisation:

TODO
