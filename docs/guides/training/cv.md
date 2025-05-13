# Cross-Validation

```sh
[uv run] darts cross-validate-smp ...
```

## Fold strategies

While cross-validating, the data can further be split into a training and validation set.
One can specify the fraction of the validation set by providing an integer to `total_folds`.
Higher values will result in smaller, validation sets and therefore more fold-combinations.
To reduce the number of folds actually run, one can provide the `n_folds` parameter to limit the number of folds actually run.
Thus, some folds will be skipped.
The "folding" is based on `scikit-learn` and currently supports the following folding methods, which can be specified by the `fold_method` parameter:

- `"kfold"`: Split the data into `total_folds` folds, where each fold can be used as a validation set. Uses [sklearn.model_selection.KFold][].
- `"stratified"`: Will use the `"empty"` column of the metadata to create `total_folds` shuffled folds where each fold contains the same amount of empty and non-empty samples. Uses [sklearn.model_selection.StratifiedKFold][].
- `"shuffle"`: Similar to `"stratified"`, but the order of the data is shuffled before splitting. Uses [sklearn.model_selection.StratifiedShuffleSplit][].
- `"region"`: Will use the `"region"` column of the metadata to create `total_folds` folds where each fold splits the data by one or multiple regions. Uses [sklearn.model_selection.GroupShuffleSplit][].
- `"region-stratified"`: Merge of the `"region"` and `"stratified"` methods. Uses [sklearn.model_selection.StratifiedGroupKFold][].

Even in normal training a single KFold split is used to split between training and validation.
This can be disabled by setting `fold_method` to `None`.
In such cases, the validation set becomes equal to the training set, meaning longer validation time and the metrics are always calculated on seen data.
This is useful for e.g. the final training of a model before deployment.

??? tip "Using DartsDataModule"

    The data splitting is implemented by the [darts_segmentation.training.data.DartsDataModule][] and can therefore be used in other settings as well.

    ::: darts_segmentation.training.data.DartsDataModule
        options:
            heading_level: 3
            members: false

## Scoring strategies

To turn the information (metrics) gathered of a single cross-validation into a useful score, we need to somehow aggregate the metrics.
In cases we are only interested in a single metric, this is easy: we can easily compute the mean.
This metric can be specified by the `scoring_metric` parameter of the cross validation.
It is also possible to use multiple metrics by specifying a list of metrics in the `scoring_metric` parameter.
This, however, makes it a little more complicated.

Multi-metric scoring is implemented as combine-then-reduce, meaning that first for each fold the metrics are combined using the specified strategy, and then the results are reduced via mean.
The combining strategy can be specified by the `multi_score_strategy` parameter.
As of now, there are four strategies implemented: `"arithmetic"`, `"geometric"`, `"harmonic"` and `"min"`.

The following visualization should help visualize how the different strategies work.
Note that the loss is interpreted as "lower is better" and has also a broader range of possible values, exceeding 1.
For the multi-metric scoring with IoU and Loss the arithmetic and geometric strategies are very instable.
The scores for very low loss values where so high that the scores needed to be clipped to the range [0, 1] for the visualization to be able to show the behaviour of these strategies.
However, especially the geometric mean shows a smoother curve than the harmonic mean for the multi-metric scoring with IoU and Recall.
This should show that the strategy should be chosen carefully and in respect to the metrics used.

|              |                                                                                                              |
| -----------: | ------------------------------------------------------------------------------------------------------------ |
|   IoU & Loss | ![Scoring strategies for JaccardIndex and Loss](../assets/score_strategies_iou_loss.png){ loading=lazy }     |
| IoU & Recall | ![Scoring strategies for JaccardIndex and Recall](../assets/score_strategies_iou_recall.png){ loading=lazy } |

??? tip "Code to reproduce the visualization"

    If you are unsure which strategy to use, you can use this code snippet to make a visualization based on your metrics:

    ```py
    import numpy as np
    import xarray as xr

    a = np.arange(0, 1, 0.01)
    a = xr.DataArray(a, dims=["a"], coords={"a": a})
    # 1 / ... indicates "lower is better" - replace it if needed
    b = np.arange(0, 2, 0.01)
    b = 1 / xr.DataArray(b, dims=["b"], coords={"b": b})

    def viz_strategies(a, b):
        harmonic = 2 / (1 / a + 1 / b)
        geometric = np.sqrt(a * b)
        arithmetic = (a + b) / 2
        minimum = np.minimum(a, b)

        harmonic = harmonic.rename("harmonic mean")
        geometric = geometric.rename("geometric mean")
        arithmetic = arithmetic.rename("arithmetic mean")
        minimum = minimum.rename("minimum")

        fig, axs = plt.subplots(1, 4, figsize=(25, 5))
        axs = axs.flatten()
        harmonic.plot(ax=axs[0])
        axs[0].set_title("Harmonic")
        geometric.plot(ax=axs[1], vmax=min(geometric.max(), 1))
        axs[1].set_title("Geometric")
        arithmetic.plot(ax=axs[2], vmax=min(arithmetic.max(), 1))
        axs[2].set_title("Arithmetic")
        minimum.plot(ax=axs[3])
        axs[3].set_title("Minimum")
        return fig

    viz_strategies(a, b).show()
    ```

Each score can be provided by either ":higher" or ":lower" to indicate the direction of the metrics.
This allows to correctly combine multiple metrics by doing 1/metric before calculation if a metric is ":lower".
If no direction is provided, it is assumed to be ":higher".
Has no real effect on the single score calculation, since only the mean is calculated there.

!!! abstract "Available metrics"

    The following metrics are visible to the scoring function:
   
    - `'train/time'`
    - `'train/device/batches_per_second'`
    - `'train/device/samples_per_second'`
    - `'train/device/flops_per_second'`
    - `'train/device/mfu'`
    - `'train/loss'`
    - `'train/Accuracy'`
    - `'train/CohenKappa'`
    - `'train/F1Score'`
    - `'train/HammingDistance'`
    - `'train/JaccardIndex'`
    - `'train/Precision'`
    - `'train/Recall'`
    - `'train/Specificity'`
    - `'val/loss'`
    - `'val/Accuracy'`
    - `'val/CohenKappa'`
    - `'val/F1Score'`
    - `'val/HammingDistance'`
    - `'val/JaccardIndex'`
    - `'val/Precision'`
    - `'val/Recall'`
    - `'val/Specificity'`
    - `'val/AUROC'`
    - `'val/AveragePrecision'`
    
    These are derived from `trainer.logged_metrics`.

### Random-state

All random state of the tuning and the cross-validation is seeded to 42.
Random state of the training can be specified through a parameter.
The cross-validation will not only cross-validates along different folds but also over different random seeds.
Thus, for a single cross-validation with 5 folds and 3 seeds, 15 runs will be executed.
