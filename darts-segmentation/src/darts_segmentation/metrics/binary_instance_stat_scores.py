"""Binary instance segmentation metrics."""

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor
from torchmetrics.classification.stat_scores import _AbstractStatScores
from torchmetrics.functional.classification.accuracy import _accuracy_reduce
from torchmetrics.functional.classification.f_beta import _binary_fbeta_score_arg_validation, _fbeta_reduce
from torchmetrics.functional.classification.precision_recall import _precision_recall_reduce
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_compute,
    _binary_stat_scores_tensor_validation,
)
from torchmetrics.utilities.plot import _AX_TYPE, _CMAP_TYPE, _PLOT_OUT_TYPE, plot_confusion_matrix

from darts_segmentation.metrics.instance_helpers import mask_to_instances, match_instances


# Implementation of torchmetric classes, following the implementation of classification metrics of torchmetrics ###
# The inheritance order is:
# Metric ->
# _AbstractStatScores ->
# BinaryInstanceStatScores ->
# [BinaryInstanceRecall, BinaryInstancePrecision, BinaryInstanceF1]
###
class BinaryInstanceStatScores(_AbstractStatScores):
    """Base class for binary instance segmentation metrics."""

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        matching_threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new instance of the BinaryInstanceStatScores metric.

        Args:
            threshold (float, optional): Threshold for binarizing the prediction.
                Has no effect if the prediction is already binarized. Defaults to 0.5.
            matching_threshold (float, optional): The threshold for matching instances. Defaults to 0.5.
            multidim_average (Literal["global", "samplewise"], optional): How the average over multiple batches is
                calculated. Defaults to "global".
            ignore_index (int | None, optional): Ignores an invalid class. Defaults to None.
            validate_args (bool, optional): Weather to validate inputs. Defaults to True.
            kwargs: Additional arguments for the Metric class, regarding compute-methods.
                Please refer to torchmetrics for more examples.

        Raises:
            ValueError: If `matching_threshold` is not a float in the [0,1] range.

        """
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index, zero_division)
            if not (isinstance(matching_threshold, float) and (0 <= matching_threshold <= 1)):
                raise ValueError(
                    f"Expected arg `matching_threshold` to be a float in the [0,1] range, but got {matching_threshold}."
                )

        self.threshold = threshold
        self.matching_threshold = matching_threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self._create_state(size=1, multidim_average=multidim_average)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the metric state.

        If the predictions are logits (not between 0 and 1), they are converted to probabilities using a sigmoid and
        then binarized using the threshold.
        If the predictions are probabilities, they are binarized using the threshold.

        Args:
            preds (Tensor): Predictions from model (logits or probabilities).
            target (Tensor): Ground truth labels.

        Raises:
            ValueError: If `preds` and `target` have different shapes.
            ValueError: If the input targets are not binary masks.

        """
        if self.validate_args:
            _binary_stat_scores_tensor_validation(preds, target, self.multidim_average, self.ignore_index)
            if not preds.dim() == 3:
                raise ValueError(f"Expected `preds` and `target` to have 3 dimensions (BHW), but got {preds.dim()}.")
            if self.ignore_index is None and target.max() > 1:
                raise ValueError(
                    "Expected binary mask, got more than 1 unique value in target."
                    " You can set 'ignore_index' to ignore a class."
                )

        # Format
        if preds.is_floating_point():
            if not torch.all((preds >= 0) * (preds <= 1)):
                # preds is logits, convert with sigmoid
                preds = preds.sigmoid()
            preds = preds > self.threshold

        if self.ignore_index is not None:
            invalid_idx = target == self.ignore_index
            preds = preds.clone()
            preds[invalid_idx] = 0  # This will prevent from counting instances in the ignored area
            target = (target == 1).to(torch.uint8)

        # Update state
        instance_list_target = mask_to_instances(target.to(torch.uint8), self.validate_args)
        instance_list_preds = mask_to_instances(preds.to(torch.uint8), self.validate_args)

        for target_i, preds_i in zip(instance_list_target, instance_list_preds):
            tp, fp, fn = match_instances(
                target_i,
                preds_i,
                match_threshold=self.matching_threshold,
                validate_args=self.validate_args,
            )
            self._update_state(tp, fp, 0, fn)

    def compute(self) -> Tensor:  # noqa: D102
        tp, fp, tn, fn = self._final_state()
        return _binary_stat_scores_compute(tp, fp, tn, fn, self.multidim_average)


class BinaryInstanceRecall(BinaryInstanceStatScores):
    """Binary instance recall metric."""

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:  # noqa: D102
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
            zero_division=self.zero_division,
        )

    def plot(  # noqa: D102
        self,
        val: Tensor | Sequence[Tensor] | None = None,
        ax: _AX_TYPE | None = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryInstancePrecision(BinaryInstanceStatScores):
    """Binary instance precision metric."""

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:  # noqa: D102
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "precision",
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
            zero_division=self.zero_division,
        )

    def plot(  # noqa: D102
        self,
        val: Tensor | Sequence[Tensor] | None = None,
        ax: _AX_TYPE | None = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryInstanceAccuracy(BinaryInstanceStatScores):
    """Binary instance accuracy metric."""

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> Tensor:  # noqa: D102
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
        )

    def plot(  # noqa: D102
        self,
        val: Tensor | Sequence[Tensor] | None = None,
        ax: _AX_TYPE | None = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryInstanceFBetaScore(BinaryInstanceStatScores):
    """Binary instance F-beta score metric."""

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        beta: float,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        """Create a new instance of the BinaryInstanceFBetaScore metric.

        Args:
            beta (float): The beta parameter for the F-beta score.
            threshold (float, optional): Threshold for binarizing the prediction.
                Has no effect if the prediction is already binarized. Defaults to 0.5.
            matching_threshold (float, optional): The threshold for matching instances. Defaults to 0.5.
            multidim_average (Literal["global", "samplewise"], optional): How the average over multiple batches is
                calculated. Defaults to "global".
            ignore_index (int | None, optional): Ignores an invalid class. Defaults to None.
            validate_args (bool, optional): Weather to validate inputs. Defaults to True.
            zero_division (float, optional): Value to return when there is a zero division. Defaults to 0.
            kwargs: Additional arguments for the Metric class, regarding compute-methods.
                Please refer to torchmetrics for more examples.

        """
        super().__init__(
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _binary_fbeta_score_arg_validation(beta, threshold, multidim_average, ignore_index, zero_division)
        self.validate_args = validate_args
        self.zero_division = zero_division
        self.beta = beta

    def compute(self) -> Tensor:  # noqa: D102
        tp, fp, tn, fn = self._final_state()
        return _fbeta_reduce(
            tp,
            fp,
            tn,
            fn,
            self.beta,
            average="binary",
            multidim_average=self.multidim_average,
            zero_division=self.zero_division,
        )

    def plot(  # noqa: D102
        self,
        val: Tensor | Sequence[Tensor] | None = None,
        ax: _AX_TYPE | None = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryInstanceF1Score(BinaryInstanceFBetaScore):
    """Binary instance F1 score metric."""

    def __init__(
        self,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        """Create a new instance of the BinaryInstanceF1Score metric.

        Args:
            threshold (float, optional): Threshold for binarizing the prediction.
                Has no effect if the prediction is already binarized. Defaults to 0.5.
            matching_threshold (float, optional): The threshold for matching instances. Defaults to 0.5.
            multidim_average (Literal["global", "samplewise"], optional): How the average over multiple batches is
                calculated. Defaults to "global".
            ignore_index (int | None, optional): Ignores an invalid class. Defaults to None.
            validate_args (bool, optional): Weather to validate inputs. Defaults to True.
            zero_division (float, optional): Value to return when there is a zero division. Defaults to 0.
            kwargs: Additional arguments for the Metric class, regarding compute-methods.
                Please refer to torchmetrics for more examples.

        """
        super().__init__(
            beta=1.0,
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            zero_division=zero_division,
            **kwargs,
        )

    def plot(  # noqa: D102
        self,
        val: Tensor | Sequence[Tensor] | None = None,
        ax: _AX_TYPE | None = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        return self._plot(val, ax)


class BinaryInstanceConfusionMatrix(BinaryInstanceStatScores):
    """Binary instance confusion matrix metric."""

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    def __init__(
        self,
        normalize: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new instance of the BinaryInstanceConfusionMatrix metric.

        Args:
            normalize (bool, optional): If True, return the confusion matrix normalized by the number of instances.
                If False, return the confusion matrix without normalization. Defaults to None.
            threshold (float, optional): Threshold for binarizing the prediction.
                Has no effect if the prediction is already binarized. Defaults to 0.5.
            matching_threshold (float, optional): The threshold for matching instances. Defaults to 0.5.
            multidim_average (Literal["global", "samplewise"], optional): How the average over multiple batches is
                calculated. Defaults to "global".
            ignore_index (int | None, optional): Ignores an invalid class. Defaults to None.
            validate_args (bool, optional): Weather to validate inputs. Defaults to True.
            kwargs: Additional arguments for the Metric class, regarding compute-methods.
                Please refer to torchmetrics for more examples.

        Raises:
            ValueError: If `normalize` is not a bool.

        """
        super().__init__(**kwargs)
        if normalize is not None and not isinstance(normalize, bool):
            raise ValueError(f"Argument `normalize` needs to be of bool type but got {type(normalize)}")
        self.normalize = normalize

    def compute(self) -> Tensor:  # noqa: D102
        tp, fp, tn, fn = self._final_state()
        # tn is always 0
        if self.normalize:
            all = tp + fp + fn
            return torch.tensor([[0, fp / all], [fn / all, tp / all]], device=tp.device)
        else:
            return torch.tensor([[tn, fp], [fn, tp]], device=tp.device)

    def plot(  # noqa: D102
        self,
        val: Tensor | None = None,
        ax: _AX_TYPE | None = None,  # type: ignore
        add_text: bool = True,
        labels: list[str] | None = None,  # type: ignore
        cmap: _CMAP_TYPE | None = None,  # type: ignore
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        val = val or self.compute()
        if not isinstance(val, Tensor):
            raise TypeError(f"Expected val to be a single tensor but got {val}")
        fig, ax = plot_confusion_matrix(val, ax=ax, add_text=add_text, labels=labels, cmap=cmap)
        return fig, ax
