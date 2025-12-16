# Sentinel-2 Data Sources and Filtering

DARTS supports multiple sources for Sentinel-2 data, each with different characteristics and filtering capabilities. This guide explains the available options and how to use them effectively.

## Available Data Sources

### CDSE Scenes (Individual Scenes)

**Source identifier**: `cdse` (default)

Individual Sentinel-2 Level-2A scenes from the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) accessed via their STAC API.

**Characteristics**:

- **Resolution**: 10m, 20m, and 60m bands (all resampled to 10m during loading)
- **Coverage**: Individual satellite passes (~100km × 100km per scene)
- **Temporal resolution**: Every 5 days (combined Sentinel-2A and 2B)
- **Processing level**: Level-2A (surface reflectance)
- **Quality information**: Scene Classification Layer (SCL) for cloud/snow masking
- **Available bands**: All Sentinel-2 bands (B01-B12, plus SCL)

**Best for**:

- High temporal resolution analysis
- Cloud-free scene selection
- Specific date requirements
- Small to medium study areas

**Example**:
```toml
[darts.inference.sentinel2-sequential]
raw-data-source = "cdse"
aoi-file = "./data/myaoi.gpkg"
start-date = "2024-07-01"
end-date = "2024-09-30"
max-cloud-cover = 10
max-snow-cover = 5
```

### CDSE Mosaics (Quarterly Composites)

**Source identifier**: `cdse-mosaic`

Quarterly mosaic composites from CDSE's [Global Mosaics collection](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html#sentinel-2-level-3-quarterly-mosaics).

**Characteristics**:

- **Resolution**: 10m only (composite of multiple acquisitions)
- **Coverage**: Global MGRS tiles (~110km × 110km)
- **Temporal resolution**: Quarterly (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
- **Processing level**: Level-3 (composite surface reflectance)
- **Quality information**: Number of observations per pixel (used for quality masking)
- **Available bands**: RGB-NIR only (B02-Blue, B03-Green, B04-Red, B08-NIR, plus observations layer)

**Best for**:

- Large-scale processing (regional to continental)
- Seasonal analysis
- Reduced cloud interference (composites select best pixels)
- Lower storage requirements than individual scenes

**Quality Masking**:

Unlike scenes with SCL, mosaics use an observation count layer:

- **Invalid** (quality_mask = 0): observations == 0
- **Low quality** (quality_mask = 1): 1 ≤ observations ≤ 3
- **High quality** (quality_mask = 2): observations > 3

**Example**:
```toml
[darts.inference.sentinel2-sequential]
raw-data-source = "cdse-mosaic"
aoi-file = "./data/myaoi.gpkg"
quarters = [3]  # Summer quarter
years = [2024, 2025]
```

### GEE (Google Earth Engine)

**Source identifier**: `gee`

Sentinel-2 Level-2A scenes from [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED).

**Characteristics**:

- **Resolution**: 10m, 20m, and 60m bands (resampled to 10m)
- **Coverage**: Individual satellite passes
- **Temporal resolution**: Every 5 days
- **Processing level**: Level-2A (surface reflectance, harmonized)
- **Quality information**: SCL cloud/snow masking
- **Available bands**: All Sentinel-2 bands (B01-B12, plus SCL)

!!! warning "Processing inconsistencies"
    GEE's Sentinel-2 data has been loaded only once and uses different processing levels across years. This can result in spectral inconsistencies that may reduce model performance. CDSE is recommended for production use.

**Best for**:

- Users already working on Google Cloud Platform
- Integration with other GEE datasets
- Prototyping and exploration

**Example**:
```toml
[darts.inference.sentinel2-sequential]
raw-data-source = "gee"
ee-project = "my-gee-project"
aoi-file = "./data/myaoi.gpkg"
start-date = "2024-07-01"
end-date = "2024-09-30"
```

## Scene Selection Methods

Four mutually exclusive methods for selecting Sentinel-2 data:

### 1. Direct Scene IDs

Provide exact scene identifiers. Works with all sources.

```toml
scene-ids = [
    "20230701T194909_20230701T195350_T11XNA",
    "20230704T195909_20230704T200350_T11XNA"
]
```

For mosaics:
```toml
scene-ids = [
    "Sentinel-2_mosaic_2024_Q3_33UUP_0_0"
]
```

### 2. Scene ID File

Load scene IDs from a file (one per line, JSON array).

```toml
scene-id-file = "./scene_ids.json"
```

### 3. Tile IDs with Filtering

Specify MGRS tile IDs and apply temporal/quality filters.

```toml
tile-ids = ["33UUP", "33UVP"]
start-date = "2024-07-01"
end-date = "2024-09-30"
max-cloud-cover = 10
max-snow-cover = 5
```

### 4. AOI with Filtering

Use a geographic area of interest (shapefile/geopackage) with filters.

```toml
aoi-file = "./data/myaoi.gpkg"
start-date = "2024-07-01"
end-date = "2024-09-30"
max-cloud-cover = 10
max-snow-cover = 5
```

## Filtering Parameters

When using tile IDs or AOI files for scene selection, you can apply various filters to narrow down the results based on temporal and quality criteria.

### Date-Based Filtering

You can filter scenes temporally using either precise date ranges or more flexible month/quarter/year combinations, depending on your data source.

#### Date Range (CDSE/GEE scenes only)

Precise date range in YYYY-MM-DD format:

```toml
start-date = "2024-07-01"
end-date = "2024-09-30"
```

#### Month/Quarter/Year (CDSE only)

Flexible temporal selection:

**For CDSE scenes** - use months or quarters:
```toml
months = [6, 7, 8]  # June through August
# quarters = [2, 3] # this would be the equivalent to month = [4, 5, 6, 7, 8, 9]
years = [2023, 2024]
```

**For CDSE mosaics** - use quarters:
```toml
quarters = [3]  # Q3 = July-September
years = [2023, 2024, 2025]
```

!!! warning "GEE limitations"
    Month/quarter/year filtering is not supported for GEE. Use date ranges instead.

### Quality-Based Filtering

For individual scenes (not mosaics), you can filter based on cloud and snow cover percentages derived from the Scene Classification Layer.

#### Cloud Cover (CDSE/GEE scenes only)

Maximum acceptable cloud cover percentage (0-100):

```toml
max-cloud-cover = 10  # Default
```

Disabling cloud filtering:
```toml
max-cloud-cover = 100  # Accept all scenes
# Or, alternativly, leave it empty since it defaults to None
```

#### Snow Cover (CDSE/GEE scenes only)

Maximum acceptable snow cover percentage (0-100):

```toml
max-snow-cover = 5  # Default
# Or, alternativly, leave it empty since it defaults to None
```

!!! note "Mosaic quality"
    CDSE mosaics don't have cloud/snow cover metadata. Quality is determined by the number of observations per pixel instead.

## Comparison Table

| Feature                | CDSE Scenes        | CDSE Mosaics        | GEE                |
| ---------------------- | ------------------ | ------------------- | ------------------ |
| **Resolution**         | 10/20/60m          | 10m                 | 10/20/60m          |
| **Available bands**    | All (B01-B12)      | RGB-NIR only        | All (B01-B12)      |
| **Temporal**           | Every 5 days       | Quarterly           | Every 5 days       |
| **Cloud filtering**    | :white_check_mark: | :x:                 | :white_check_mark: |
| **Snow filtering**     | :white_check_mark: | :x:                 | :white_check_mark: |
| **Date range**         | :white_check_mark: | :x:                 | :white_check_mark: |
| **Month filtering**    | :white_check_mark: | :white_check_mark:* | :x:                |
| **Quarter filtering**  | :x:                | :white_check_mark:  | :x:                |
| **Quality mask**       | SCL-based          | Observation count   | SCL-based          |
| **Storage efficiency** | Low                | High                | Low                |
| **Processing speed**   | Medium             | Fast                | Slow (network)     |
| **Best for**           | Precise dates      | Large areas         | GCP workflows      |

\* Automatically converted to quarters

## Usage Examples

The following examples demonstrate common use cases for different Sentinel-2 data sources and filtering approaches.

### Example 1: Summer Analysis with CDSE Scenes

High-quality summer scenes with strict cloud filtering:

```toml
[darts.inference.sentinel2-sequential]
raw-data-source = "cdse"
aoi-file = "./data/study_area.gpkg"
start-date = "2024-06-15"
end-date = "2024-08-31"
max-cloud-cover = 5
max-snow-cover = 0
```

### Example 2: Multi-Year Quarterly Mosaics

Large-scale seasonal analysis using mosaics:

```toml
[darts.inference.sentinel2-sequential]
raw-data-source = "cdse-mosaic"
tile-ids = ["33UUP", "33UVP", "33UWP"]
quarters = [2, 3]  # Spring and summer
years = [2022, 2023, 2024]
```

### Example 3: Specific Tiles with Month Filtering

CDSE scenes for specific months across multiple years:

```toml
[darts.inference.sentinel2-sequential]
raw-data-source = "cdse"
tile-ids = ["33UUP"]
months = [7, 8]  # July and August
years = [2020, 2021, 2022, 2023, 2024]
max-cloud-cover = 15
```

### Example 4: Google Earth Engine Workflow

Using GEE with date range (required for GEE):

```toml
[darts.inference.sentinel2-sequential]
raw-data-source = "gee"
ee-project = "my-gee-project"
aoi-file = "./data/study_area.gpkg"
start-date = "2024-07-01"
end-date = "2024-07-31"
max-cloud-cover = 10
```

## Best Practices

Follow these guidelines to optimize your Sentinel-2 data acquisition and processing workflow.

### Choosing a Data Source

1. **Default choice**: Use `cdse` for most applications
2. **Large-scale processing**: Use `cdse-mosaic` for regions > 1000 km²
3. **Seasonal analysis**: Use `cdse-mosaic` with quarters
4. **Specific dates needed**: Use `cdse` or `gee` with date ranges
5. **GCP integration**: Use `gee` only if already on Google Cloud

### Filtering Strategy

1. **Start conservative**: Begin with low cloud/snow thresholds (5-10%)
2. **Iterate if needed**: Increase thresholds if insufficient scenes found
3. **Check coverage**: Use `prep-data` to verify scene availability before full pipeline runs

### Download Optimization

1. **Enable caching**: Use `raw-data-store` to avoid downloading the same data again and again (enabled by default)
2. **Disable for one-off**: Use `--no-raw-data-store` for single-use processing

## Related Documentation

- [Pipeline v2 Guide](pipeline-v2.md): Complete pipeline documentation
- [Offline Processing](offline.md): Pre-downloading data for offline use
- [Path Management](paths.md): Configuring data storage locations
- [Components Reference](components.md): Low-level API documentation
