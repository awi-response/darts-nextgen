# How a model is born

!!! tip "Recommendations & best practices"

    It is recommended to use a terminal multiplexer like `tmux`, `screen` or `zellij` to run multiple training runs in parallel.
    This way there is no need to have multiple terminal open over the span of multiple days.

    Using the folding method `"region-stratified"` enables more data-efficient training because the test split can then be set to random.

    Multi-scoring strategy should be `"geometric"` for combinations of recall, precision, f1 and jaccard index / iou, else `"harmonic"`.

To do a super quick tutorial to get you started go the [quickstart](quickstart.md) guide.

The "model creation process" (training) is implemented as a three-level hierarchy, meaning the upper level does several calls to the level below it:

1. [Tuning (Hyperparameter-Sweeps)](tune.md)
2. [Cross-Validation](cv.md)
3. [(Training-) Run](training.md)

![Training Process](../../assets/tune-cv-train.png)

## Artifacts and Naming

Training artefacts are stored in an organized way so that one does not get lost in 1000s of different directories.
Especially when tuning hyperparameters, a lot of different runs are created, which can be difficult to track.

For organisation, each tune, cv and training run has it's own name, which _can_ be provided manually, but is usually generated automatically.
The name can only be provided manually for the call-level - hence when tuning one can only provide the name for the tune, respective cross-validations and training runs are named automatically based on the provided name.
If no name is provided, a random, but human-readable name is generated.
Further, a random 8-character id is also generated for each run, primarily for tracking purposes with Weights & Biases.

The naming scheme is as follows:

- `tune_name`: automatically generated or provided
- `cv_name`: `{tune_name}-cv{hp_index}` if called by tune, else automatically generated or provided
- `run_name`: `{cv_name}-run-f{fold_index}s{seed}` if called by cross-validation (or indirect tune), else automatically generated or provided
- `run_id`: 8-character id

Artifacts are stored in the following hierarchy:

- Created by runs of tunes: `{artifact_dir}/{tune_name}/{cv_name}/{run_name}-{run_id}`
- Created by runs of cross-validations: `{artifact_dir}/_cross_validations/{cv_name}/{run_name}-{run_id}`
- Created by single runs: `{artifact_dir}/_runs/{run_name}-{run_id}`

This way, the top-level `artifact_dir` is kept clean and organized.

!!! tip "Local vs. WandB"

    The training uses a local directory for storing the artifacts, such as metrics or the model.
    The final directory where these artifacts is always called `{run_name}-{run_id}`.
    This way, it should be easy to relate which artifacts belong to which run in wandb, where the url of a run is always `https://wandb.ai/{wandb_entity}/{wandb_project}/runs/{run_id}`.

The cross-validation will not only contain the artifacts from the training runs but also a `run_infos.parquet` file with information about each run / experiment.
This dataframe contains a `fold`, `seed`, `duration`, `checkpoint`, `is_unstable`, `is_unstable` and metrics columns, where the metrics are taken from the `trainer.callback_metrics`.
The `is_unstable` column indicates whether the score-metrics of the run were unstable (not finite or zero).
Further, it also contains a `score` and a `score_is_unstable` column, which contains the score and a boolean indicating whether any run of the cross-validation was unstable.
These columns contain the same value for every row (run), since they are valid for the complete cross-validation.

Weights & Biases is optionally used for further tracking and logging.
`wandb_project` and `wandb_entity` can be used to specify the project and entity for logging.
Wandb will create a run `run_id` named `{run_name}`, meaning the id can be used to directly access the run via link and the name can be used for searching a run.
For cross-validation and tuning `cv_name` and `tune_name` are set as `job_type` and `group` to emulate sweeps.
This is a workaround and could potentially fixed if wandb will update their client library to allow the manual creation of sweeps.
