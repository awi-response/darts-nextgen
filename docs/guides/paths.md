# Paths and Data Management

DARTS uses a structured approach to manage data storage locations across different storage types (fast SSD storage vs. large/slow storage). This guide explains how the path system works and how to configure it.

## Overview

DARTS organizes data into two main storage categories:

- **Fast Storage**: For data requiring quick access (training data, models)
- **Large Storage**: For large datasets and outputs (auxiliary data, artifacts, cache, logs, outputs)

## Directory Structure

### Fast Storage Directories

- `training/`: Training datasets
- `models/`: Trained model files (*.pt)

### Large Storage Directories

- `aux/`: Auxiliary data (ArcticDEM, TCVis)
- `artifacts/`: Training artifacts and checkpoints
- `cache/`: Temporary cache files
- `logs/`: Log files
- `output/`: Pipeline output results
- `input/`: Input data (Planet, Sentinel-2)
- `archive/`: Archived data

### Auxiliary Data Subdirectories

Within the `aux/` directory:

- `admin_boundaries/`: Administrative boundaries and regions
- `arcticdem_2m.icechunk/`: ArcticDEM at 2m resolution
- `arcticdem_10m.icechunk/`: ArcticDEM at 10m resolution
- `arcticdem_32m.icechunk/`: ArcticDEM at 32m resolution
- `tcvis.icechunk/`: Temporal Change Visualization data

### Input Data Subdirectories

Within the `input/` directory:

- `planet/tiles/`: PlanetScope orthotiles
- `planet/scenes/`: PlanetScope scenes
- `sentinel2/grid/`: Sentinel-2 grid data
- `sentinel2/cdse-scenes/`: Sentinel-2 raw data from Copernicus Data Space Ecosystem
- `sentinel2/gee-scenes/`: Sentinel-2 raw data from Google Earth Engine

## Configuration Methods

DARTS provides three ways to configure paths, listed in priority order (highest to lowest):

### 1. CLI Flags (Highest Priority)

You can specify custom paths using CLI flags when running any DARTS command. These flags override all other settings.

#### Default Directory Flags

These flags set the base directories for fast and large storage:

```bash
darts inference planet-sequential \
  --default-dirs.darts-dir /path/to/data \
  --default-dirs.fast-dir /path/to/fast/storage \
  --default-dirs.large-dir /path/to/large/storage \
  ...
```

- `--default-dirs.darts-dir`: Sets both fast and large directories (if they're not specified separately)
- `--default-dirs.fast-dir`: Overrides the fast storage location
- `--default-dirs.large-dir`: Overrides the large storage location

#### Specific Directory Flags

You can also override specific directories:

```bash
darts inference sentinel2-sequential \
  --output-data-dir /custom/output \
  --arcticdem-dir /custom/arcticdem \
  --tcvis-dir /custom/tcvis \
  ...
```

Available specific directory flags:

- `--output-data-dir`: Where to write pipeline outputs
- `--arcticdem-dir`: Location of ArcticDEM data
- `--tcvis-dir`: Location of TCVis data

### 2. Environment Variables (Medium Priority)

Set environment variables before running DARTS:

```bash
export DARTS_DATA_DIR=/path/to/data
export DARTS_FAST_DATA_DIR=/path/to/fast/storage
export DARTS_LARGE_DATA_DIR=/path/to/large/storage

darts inference planet-sequential ...
```

Environment variables:

- `DARTS_DATA_DIR`: Base directory (used if fast/large not specified)
- `DARTS_FAST_DATA_DIR`: Fast storage location
- `DARTS_LARGE_DATA_DIR`: Large storage location

### 3. Configuration Files (Lowest Priority)

Specify paths in your `config.toml` file:

```toml
[default_dirs]
darts_dir = "/path/to/data"
fast_dir = "/path/to/fast/storage"
large_dir = "/path/to/large/storage"

# Or override specific directories
output_data_dir = "/custom/output"
arcticdem_dir = "/custom/arcticdem"
```

Then run with the config file:

```bash
darts --config-file config.toml inference planet-sequential ...
```

### 4. Default Behavior (No Configuration)

If no paths are configured, DARTS uses the current working directory for all storage locations.

## Path Resolution Priority

When DARTS resolves paths, it follows this priority:

1. **Explicit CLI flag** (e.g., `--output-data-dir`)
2. **Environment variable** (e.g., `DARTS_LARGE_DATA_DIR`)
3. **Config file value**
4. **Current working directory**

For `fast_dir` and `large_dir`:

- If not set, they default to `darts_dir`
- If `darts_dir` is not set, they default to the current working directory

## Common Usage Examples

### Example 1: Simple Setup (Same Location)

Use the current directory for everything:

```bash
cd /data/darts-project
darts inference planet-sequential --image-ids IMG_001 IMG_002
```

### Example 2: Split Fast/Large Storage

Store training data and models on SSD, everything else on HDD:

```bash
darts inference sentinel2-sequential \
  --default-dirs.fast-dir /ssd/darts-fast \
  --default-dirs.large-dir /hdd/darts-large \
  --tile-ids 33UXP 33UYP
```

### Example 3: Custom Output Location

Override just the output directory:

```bash
darts inference planet-sequential \
  --output-data-dir /project/results \
  --image-ids IMG_001
```

### Example 4: Shared Auxiliary Data

Use shared auxiliary data with local outputs:

```bash
darts inference sentinel2-sequential \
  --default-dirs.large-dir /local/output \
  --arcticdem-dir /shared/arcticdem_10m.icechunk \
  --tcvis-dir /shared/tcvis.icechunk \
  --tile-ids 33UXP
```

### Example 5: Using Environment Variables

Set up your environment once:

```bash
export DARTS_FAST_DATA_DIR=/ssd/darts
export DARTS_LARGE_DATA_DIR=/hdd/darts

# All subsequent commands use these paths
darts inference planet-sequential --image-ids IMG_001
darts training train-smp --config train.toml
```

### Example 6: Configuration File for Projects

Create a project-specific `config.toml`:

```toml
[default_dirs]
fast_dir = "/project/fast-storage"
large_dir = "/project/large-storage"

[pipeline]
output_data_dir = "/project/results"
overwrite = false
offline = true
```

Run with:

```bash
darts --config-file project-config.toml inference planet-sequential --image-ids IMG_001
```

## Debugging Paths

To see which paths DARTS is using, use the `debug-paths` command:

```bash
# Check default paths
darts debug-paths

# Check paths with custom settings
darts debug-paths --default-dirs.fast-dir /ssd/darts --default-dirs.large-dir /hdd/darts

# See what environment variables would set
export DARTS_FAST_DATA_DIR=/ssd/darts
darts debug-paths
```

This will print all resolved paths:

- Fast Directory
- Large Directory  
- Auxiliary Directory
- Artifacts Directory
- Training Directory
- Cache Directory
- Logs Directory
- Output Directory
- Models Directory
- Input Directory
- Archive Directory
- And all subdirectories

## Best Practices

1. **Use Environment Variables for Persistent Settings**: Set `DARTS_FAST_DATA_DIR` and `DARTS_LARGE_DATA_DIR` in your shell profile for consistent paths across sessions.

2. **Use Config Files for Projects**: Create project-specific config files that team members can share.

3. **Use CLI Flags for One-Off Changes**: Override paths temporarily without changing your environment or config files.

4. **Keep Auxiliary Data Centralized**: Store ArcticDEM and TCVis in a shared location to avoid duplicating large datasets.

5. **Organize by Project**: Structure your large storage with project subdirectories:

   ```text
   /large/darts/
   ├── project-A/output/
   ├── project-B/output/
   └── shared/aux/
   ```

6. **Check Before Large Runs**: Use `darts debug-paths` to verify your configuration before starting large processing jobs.

## Pipeline-Specific Considerations

### Training Pipelines

Training data is automatically organized by pipeline and patch size:

- `{fast_dir}/training/{pipeline}_{patch_size}/`

Example: `training/planet_256/` for Planet pipeline with 256x256 patches.

### Inference Pipelines

- **Input Data**: Automatically discovered in `{large_dir}/input/`
- **Models**: Automatically discovered in `{fast_dir}/models/` (all *.pt files)
- **Output**: Written to `{large_dir}/output/` by default

### Data Preparation

Use the `prep-data` commands to download data for offline use:

```bash
# Prepare optical and auxiliary data
darts inference prep-data planet \
  --default-dirs.large-dir /hdd/darts \
  --pipeline.image-ids IMG_001 IMG_002 \
  --aux

# Prepare only optical data
darts inference prep-data sentinel2 \
  --pipeline.tile-ids 33UXP \
  --optical
```

## Troubleshooting

### Issue: "Cannot find models"

**Solution**: Ensure model files (*.pt) are in `{fast_dir}/models/` or specify explicitly:

```bash
--model-files /path/to/model.pt
```

### Issue: "Permission denied"

**Solution**: Check directory permissions and ensure the user has write access:

```bash
chmod -R u+w /path/to/darts-data
```

### Issue: "Disk full" during training

**Solution**: Ensure fast storage has sufficient space, or redirect training data:

```bash
--default-dirs.fast-dir /larger/drive/darts-fast
```

### Issue: "Path not found"

**Solution**: DARTS creates directories automatically, but parent directories must exist:

```bash
mkdir -p /parent/path
darts inference ... --default-dirs.large-dir /parent/path/darts-large
```

## Programmatic Usage

When using DARTS as a Python library:

```python
from darts_utils.paths import paths, DefaultPaths

# Set paths programmatically
paths.set_defaults(DefaultPaths(
    fast_dir="/ssd/darts",
    large_dir="/hdd/darts"
))

# Access paths
model_dir = paths.models
output_dir = paths.out
arcticdem_10m = paths.arcticdem(10)

# Use in your code
my_output = output_dir / "my_results"
```

