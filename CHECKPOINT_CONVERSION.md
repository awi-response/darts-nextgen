# Consistency Model Checkpoint Conversion

## Problem

When loading consistency model checkpoints in DARTS, PyTorch Lightning's `load_from_checkpoint()` fails with:
```
ModuleNotFoundError: No module named 'src'
```

**Root cause:** The checkpoint file contains pickled references to the original module structure (`src.consistency_model.*`). When unpickling in DARTS (which uses `darts_superresolution.*` module paths), Python cannot find those modules.

## Solution

Convert the checkpoint once to remove misleading pickle metadata. The converted checkpoint contains only the model weights and hyperparameters, making it portable across module structures.

## How to Convert

### Option 1: Using the helper script (recommended)
From the darts-nextgen repo root:
```bash
cd darts-superresolution
./convert_checkpoint.sh /path/to/original.ckpt [/path/to/output.ckpt]
```

Example with your checkpoint:
```bash
cd darts-superresolution
./convert_checkpoint.sh /p/scratch/hai_earth_04/lucas/Consistency_Model/checkpoint/consistency_wavelet0.3_image0.7_10steps_no_l1_1.0lpips_bins1.0_continued-epoch=169-val_loss=0.0014.ckpt
```

### Option 2: Manual conversion from consistency repo
```bash
cd consistency_model_distillation_for_sr3
python convert_checkpoint.py /path/to/original.ckpt /path/to/output.ckpt
```

## What Gets Converted

The conversion function:
1. **Loads** the checkpoint using the original consistency repo environment (src.* imports work there)
2. **Extracts** only the model weights (`state_dict`) and hyperparameters
3. **Saves** as a clean dict with no embedded module references

**Original checkpoint structure:**
```python
{
    # Contains pickled ConsistencyWavelet class with src.* references
    # → causes unpickling to fail in DARTS
}
```

**Converted checkpoint structure:**
```python
{
    'state_dict': {...},           # Model weights (portable)
    'hyper_parameters': {...},     # Training config
}
```

## How DARTS Loads It

In `darts-superresolution/src/darts_superresolution/upscale.py`, the `_load_consistency_model()` method now:

1. **Tries converted format first** (dict with `state_dict` key)
   - Fast, clean, no pickle issues
   - Creates fresh ConsistencyWavelet instance
   - Loads weights into it

2. **Falls back to original format** (PyTorch Lightning checkpoint)
   - Supports old checkpoints temporarily
   - Shows helpful error with conversion instructions if it fails

## Next Steps

1. Convert your checkpoint:
   ```bash
   cd darts-nextgen/darts-superresolution
   ./convert_checkpoint.sh /p/scratch/hai_earth_04/lucas/Consistency_Model/checkpoint/consistency_wavelet0.3_image0.7_10steps_no_l1_1.0lpips_bins1.0_continued-epoch=169-val_loss=0.0014.ckpt
   ```

2. Update `superresolution_test.py` to use the converted checkpoint:
   ```python
   model = Sentinel2Upscaler(
       model_path="/path/to/consistency_wavelet0.3_image0.7_10steps_no_l1_1.0lpips_bins1.0_continued-epoch=169-val_loss=0.0014.converted.ckpt",
       backend="consistency",
   )
   ```

3. Run the test:
   ```bash
   python superresolution_test.py
   ```

## Technical Details

- **Conversion location:** `consistency_model_distillation_for_sr3/convert_checkpoint.py`
- **Consumer code:** `darts-nextgen/darts-superresolution/src/darts_superresolution/upscale.py:_load_consistency_model()`
- **Format specification:** `torch.save()` with plain dict (no custom classes)
- **Compatibility:** Works across any Python environment with PyTorch

## Benefits

✅ One-time operation (checkpoint converted once)
✅ No runtime sys.path manipulation
✅ No pickle compatibility layer needed
✅ Portable across different DARTS deployments
✅ Clean error messages if original checkpoint is used by mistake
