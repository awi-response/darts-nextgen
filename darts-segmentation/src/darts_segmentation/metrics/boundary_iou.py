"""Boundary IoU metric for binary segmentation tasks."""

from typing import Literal, TypedDict, Unpack

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_tensor_validation,
)

from darts_segmentation.metrics.boundary_helpers import get_boundary

MatchingMetric = Literal["iou", "boundary"]


class BinaryBoundaryIoUKwargs(TypedDict):
    """Keyword arguments for the BinaryBoundaryIoU metric."""

    zero_division: Literal[0, 1]
    compute_on_cpu: bool
    dist_sync_on_step: bool
    process_group: str
    dist_sync_fn: callable
    distributed_available_fn: callable
    sync_on_compute: bool
    compute_with_cache: bool


class BinaryBoundaryIoU(Metric):
    """Binary Boundary IoU metric for binary segmentation tasks.

    This metric is similar to the Binary Intersection over Union (IoU or Jaccard Index) metric,
    but instead of comparing all pixels it only compares the boundaries of each foreground object.
    """

    intersection: Tensor | list[Tensor]
    union: Tensor | list[Tensor]

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        dilation: float | int = 0.02,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Unpack[BinaryBoundaryIoUKwargs],
    ):
        """Create a new instance of the BinaryBoundaryIoU metric.

        Please see the
        [torchmetrics docs](https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs)
        for more info about the **kwargs.

        Args:
            dilation (float | int, optional): The dilation (factor) / width of the boundary.
                Dilation in pixels if int, else ratio to calculate `dilation = dilation_ratio * image_diagonal`.
                Default: 0.02
            threshold (float, optional): Threshold for binarizing the prediction.
                Has no effect if the prediction is already binarized. Defaults to 0.5.
            multidim_average (Literal["global", "samplewise"], optional): How the average over multiple batches is
                calculated. Defaults to "global".
            ignore_index (int | None, optional): Ignores an invalid class.  Defaults to None.
            validate_args (bool, optional): Weather to validate inputs. Defaults to True.

        Keyword Args:
            zero_division (int):
                Value to return when there is a zero division. Default is 0.
            compute_on_cpu (bool):
                If metric state should be stored on CPU during computations. Only works for list states.
            dist_sync_on_step (bool):
                If metric state should synchronize on ``forward()``. Default is ``False``.
            process_group (str):
                The process group on which the synchronization is called. Default is the world.
            dist_sync_fn (callable):
                Function that performs the allgather option on the metric state. Default is a custom
                implementation that calls ``torch.distributed.all_gather`` internally.
            distributed_available_fn (callable):
                Function that checks if the distributed backend is available. Defaults to a
                check of ``torch.distributed.is_available()`` and ``torch.distributed.is_initialized()``.
            sync_on_compute (bool):
                If metric state should synchronize when ``compute`` is called. Default is ``True``.
            compute_with_cache (bool):
                If results from ``compute`` should be cached. Default is ``True``.

        Raises:
            ValueError: If dilation is not a float or int.

        """
        zero_division = kwargs.pop("zero_division", 0)
        super().__init__(**kwargs)

        if validate_args:
            _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index, zero_division)
            if not isinstance(dilation, float | int):
                raise ValueError(f"Expected argument `dilation` to be a float or int, but got {dilation}.")

        self.dilation = dilation
        self.threshold = threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        if multidim_average == "samplewise":
            self.add_state("intersection", default=[], dist_reduce_fx="cat")
            self.add_state("union", default=[], dist_reduce_fx="cat")
        else:
            self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("union", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the metric state.

        If the predictions are logits (not between 0 and 1), they are converted to probabilities using a sigmoid and
        then binarized using the threshold.
        If the predictions are probabilities, they are binarized using the threshold.

        Args:
            preds (Tensor): Predictions from model (logits or probabilities).
            target (Tensor): Ground truth labels.

        Raises:
            ValueError: If the input arguments are invalid.
            ValueError: If the input shapes are invalid.

        """
        if self.validate_args:
            _binary_stat_scores_tensor_validation(preds, target, self.multidim_average, self.ignore_index)
            if not preds.shape == target.shape:
                raise ValueError(
                    f"Expected `preds` and `target` to have the same shape, but got {preds.shape} and {target.shape}."
                )
            if not preds.dim() == 3:
                raise ValueError(f"Expected `preds` and `target` to have 3 dimensions, but got {preds.dim()}.")

        # Format
        if preds.is_floating_point():
            if not torch.all((preds >= 0) * (preds <= 1)):
                # preds is logits, convert with sigmoid
                preds = preds.sigmoid()
            preds = preds > self.threshold

        target = target.to(torch.uint8)
        preds = preds.to(torch.uint8)

        target_boundary = get_boundary((target == 1).to(torch.uint8), self.dilation, self.validate_args)
        preds_boundary = get_boundary(preds, self.dilation, self.validate_args)

        intersection = target_boundary & preds_boundary
        union = target_boundary | preds_boundary

        if self.ignore_index is not None:
            # Important that this is NOT the boundary, but the original mask
            valid_idx = target != self.ignore_index
            intersection &= valid_idx
            union &= valid_idx

        intersection = intersection.sum().item()
        union = union.sum().item()

        if self.multidim_average == "global":
            self.intersection += intersection
            self.union += union
        else:
            self.intersection.append(intersection)
            self.union.append(union)

    def compute(self) -> Tensor:
        """Compute the metric.

        Returns:
            Tensor: The computed metric.

        """
        if self.multidim_average == "global":
            return self.intersection / self.union
        else:
            self.intersection = torch.tensor(self.intersection)
            self.union = torch.tensor(self.union)
            return self.intersection / self.union
