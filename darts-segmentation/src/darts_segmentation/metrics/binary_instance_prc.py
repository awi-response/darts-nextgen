"""Complex binary instance segmentation metrics."""

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.average_precision import _binary_average_precision_compute
from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_tensor_validation,
)
from torchmetrics.utilities.compute import _auc_compute_without_check
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve

from darts_segmentation.metrics.instance_helpers import mask_to_instances, match_instances

MatchingMetric = Literal["iou", "boundary"]


# Implementation of torchmetric classes, following the implementation of classification metrics of torchmetrics ###
# The inheritance order is:
# Metric ->
# BinaryInstancePrecisionRecallCurve ->
# [BinaryInstanceAUROC, BinaryROC]
###


class BinaryInstancePrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for binary instance segmentation.

    This metric works similar to `torchmetrics.classification.PrecisionRecallCurve`, with two key differences:
    1. It calculates the tp, fp, fn values for each instance (blob) in the batch, and then aggregates them.
        Instead of calculating the values for each pixel.
    2. The "thresholds" argument is required.
        Calculating the thresholds at the compute stage would cost to much memory for this usecase.

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    preds: list[Tensor]
    target: list[Tensor]
    confmat: Tensor
    thesholds: Tensor

    def __init__(
        self,
        thresholds: int | list[float] | Tensor = None,
        matching_threshold: float = 0.5,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new instance of the BinaryInstancePrecisionRecallCurve metric.

        Args:
            thresholds (int | list[float] | Tensor, optional): The thresholds to use for the curve. Defaults to None.
            matching_threshold (float, optional): The threshold for matching instances. Defaults to 0.5.
            ignore_index (int | None, optional): Ignores an invalid class. Defaults to None.
            validate_args (bool, optional): Weather to validate inputs. Defaults to True.
            kwargs: Additional arguments for the Metric class, regarding compute-methods.
                Please refer to torchmetrics for more examples.

        Raises:
            ValueError: If thresholds is None.

        """
        super().__init__(**kwargs)
        if validate_args:
            _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
            if not (isinstance(matching_threshold, float) and (0 <= matching_threshold <= 1)):
                raise ValueError(
                    f"Expected arg `matching_threshold` to be a float in the [0,1] range, but got {matching_threshold}."
                )
            if thresholds is None:
                raise ValueError("Argument `thresholds` must be provided for this metric.")

        self.matching_threshold = matching_threshold
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        self.register_buffer("thresholds", thresholds, persistent=False)
        self.add_state("confmat", default=torch.zeros(len(thresholds), 2, 2, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states.

        Args:
            preds (Tensor): The predicted mask. Shape: (batch_size, height, width)
            target (Tensor): The target mask. Shape: (batch_size, height, width)

        Raises:
            ValueError: If preds and target have different shapes.
            ValueError: If the input targets are not binary masks.

        """
        if self.validate_args:
            _binary_precision_recall_curve_tensor_validation(preds, target, self.ignore_index)
            if not preds.dim() == 3:
                raise ValueError(f"Expected `preds` and `target` to have 3 dimensions (BHW), but got {preds.dim()}.")
            if self.ignore_index is None and target.max() > 1:
                raise ValueError(
                    "Expected binary mask, got more than 1 unique value in target."
                    " You can set 'ignore_index' to ignore a class."
                )

        # Format
        if not torch.all((preds >= 0) * (preds <= 1)):
            preds = preds.sigmoid()

        if self.ignore_index is not None:
            target = (target == 1).to(torch.uint8)

        instance_list_target = mask_to_instances(target.to(torch.uint8), self.validate_args)

        len_t = len(self.thresholds)
        confmat = self.thresholds.new_zeros((len_t, 2, 2), dtype=torch.int64)
        for i in range(len_t):
            preds_i = preds >= self.thresholds[i]

            if self.ignore_index is not None:
                invalid_idx = target == self.ignore_index
                preds_i = preds_i.clone()
                preds_i[invalid_idx] = 0  # This will prevent from counting instances in the ignored area

            instance_list_preds_i = mask_to_instances(preds_i.to(torch.uint8), self.validate_args)
            for target_i, preds_i in zip(instance_list_target, instance_list_preds_i):
                tp, fp, fn = match_instances(
                    target_i,
                    preds_i,
                    match_threshold=self.matching_threshold,
                    validate_args=self.validate_args,
                )
                confmat[i, 1, 1] += tp
                confmat[i, 0, 1] += fp
                confmat[i, 1, 0] += fn
        self.confmat += confmat

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:  # noqa: D102
        return _binary_precision_recall_curve_compute(self.confmat, self.thresholds)

    def plot(  # noqa: D102
        self,
        curve: tuple[Tensor, Tensor, Tensor] | None = None,
        score: Tensor | bool | None = None,
        ax: _AX_TYPE | None = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        curve_computed = curve or self.compute()
        # switch order as the standard way is recall along x-axis and precision along y-axis
        curve_computed = (curve_computed[1], curve_computed[0], curve_computed[2])

        score = (
            _auc_compute_without_check(curve_computed[0], curve_computed[1], direction=-1.0)
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed, score=score, ax=ax, label_names=("Recall", "Precision"), name=self.__class__.__name__
        )


class BinaryInstanceAveragePrecision(BinaryInstancePrecisionRecallCurve):
    """Compute the average precision for binary instance segmentation."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:  # type: ignore[override]  # noqa: D102
        return _binary_average_precision_compute(self.confmat, self.thresholds)

    def plot(  # type: ignore[override]  # noqa: D102
        self,
        val: Tensor | Sequence[Tensor] | None = None,
        ax: _AX_TYPE | None = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)
