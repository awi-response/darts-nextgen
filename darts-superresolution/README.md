# DARTS superresolution

Image superresolution of Sentinel 2 imagery for the DARTS dataset.

In this package, we can plug in a superresolution model which transforms images from Sentinel-2 native resolution (10m) in the four optical bands (R, G, B, NIR) to the same bands in a higher resolution at roughly 3m resolution. The models were trained on Sentinel-2 and PlanetScope data from the Arctic Permafrost Dataset.

# Usage

The `infer.py` entrypoint reads defaults from `src/darts_superresolution/config/config.py`, and you can override them with shell flags.

Run with defaults:

`cd darts-nextgen/darts-superresolution/src/darts_superresolution`

`uv run ./infer.py`

Run with overrides (example):

`uv run ./infer.py --batch-size 12 --backend diffusion --use-ddim --diffusion-steps 30`

Other common flags:

`--model-checkpoint /path/to/model.ckpt`

`--test-scene-dir /path/to/s2_scene_folder`

`--output-path /path/to/output.tif`

`--input-patch-size 120 --output-patch-size 384 --patch-stride 110`

`--no-input-min-max` or `--input-min-max -1 1`

`--diffusion-steps 30` (primary diffusion sampling step flag)

Normalization note for new users:

Set `inference_input_min_max` in `src/darts_superresolution/config/config.py` (`RuntimeConfig`) as the default source of truth.
The same setting can be overridden per run with `--input-min-max` / `--no-input-min-max`.

## Pipeline Integration (Near One-Liner)

From another Python module in darts-nextgen, you can call inference directly:

```python
from darts_superresolution.infer import run_inference

output_path = run_inference(
	model_checkpoint="/path/to/model.ckpt",
	test_scene_dir="/path/to/s2_scene",
	output_path="/path/to/output.tif",
	batch_size=12,
	backend="diffusion",
)
```

This keeps the orchestration code minimal while still allowing per-call overrides.

## Adding A New Backend Model

Use this checklist when integrating a new superresolution model so it works like the existing `diffusion` and `consistency` backends.

1. Add model implementation in `src/darts_superresolution/model/`.
	- Include architecture code and any backend-specific inference wrapper.
	- Keep checkpoint loading requirements explicit (for example: expected key names, tensor-only vs Lightning format).

2. Define model-level defaults in config.
	- Put architecture defaults and loader schema in `src/darts_superresolution/config/model_parameters.py`.
	- If the backend has many unique architecture settings, create a dedicated defaults module and import it from there.

3. Add runtime inference options in `src/darts_superresolution/config/config.py`.
	- Add a dataclass section such as `NewModelInferenceConfig`.
	- Add any backend selection and sampling/runtime controls here (not in the test script).

4. Register backend loading and inference in `src/darts_superresolution/util/upscale.py`.
	- Extend `backend` choices in `Sentinel2Upscaler`.
	- Add a loader method (for example `_load_new_model`) to instantiate and load checkpoints.
	- Add inference routing in `_infer_batch` so the new backend runs through the same patching/stitching pipeline.

5. Keep `src/darts_superresolution/infer.py` thin.
	- Read from `InferenceConfig`.
	- Pass backend and backend-specific options into `Sentinel2Upscaler`.
	- Avoid embedding backend constants directly in the script.

## Notes On Consistency Checkpoints

- `convert_checkpoint.py` handles checkpoint serialization compatibility (removes fragile `src.*` pickle dependencies).
- The runtime still needs a consistency config schema/class to instantiate the model before loading state dict weights.
- In this package, that role is handled by `ConsistencyConfig` in `src/darts_superresolution/config/model_parameters.py`.

