"""Scoring calculation."""

from math import isfinite
from statistics import geometric_mean, harmonic_mean, mean
from typing import Literal


def score_from_single_run(
    run_info: dict[str, float],
    scoring_metric: list[str] | str,
    multi_score_strategy: Literal["harmonic", "arithmetic", "geometric", "min"] = "harmonic",
) -> float:
    """Calculate a score from run metrics.

    Each metric can be provided by either ":higher" or ":lower" to indicate the direction of the metrics.
    This allows to correctly combine multiple metrics by doing 1/metric before calculation if a metric is ":lower".
    If no direction is provided, it is assumed to be ":higher".
    Has no real effect on the single score calculation, since only the mean is calculated there.

    In a multi-score setting, the score is calculated by combining the metrics through the specified strategy.
    Please refer to the documentation to understand the different multi-score strategies.

    Unstable runs when multi_score_strategy is "harmonic" or "geometric" will result in a score of 0.
    An unstable run is where one of the metrics is not finite or zero.

    Args:
        run_info (dict[str, float]): Dictionary containing run information and metrics
        scoring_metric (list[str] | str): Metric(s) to use for scoring.
        multi_score_strategy (Literal["harmonic", "arithmetic", "geometric", "min"], optional):
            Strategy for combining multiple metrics. Defaults to "harmonic".

    Returns:
        float: The calculated score

    Raises:
        ValueError: If an unknown multi-score strategy is provided.

    """
    if isinstance(scoring_metric, str):
        return run_info[scoring_metric]

    scores = [run_info[metric] for metric in scoring_metric]
    is_unstable = check_score_is_unstable(run_info, scoring_metric)
    if is_unstable and multi_score_strategy in ["harmonic", "geometric"]:
        return 0.0
    match multi_score_strategy:
        case "harmonic":
            return harmonic_mean(scores)
        case "arithmetic":
            return mean(scores)
        case "geometric":
            return geometric_mean(scores)
        case "min":
            return min(scores)
        case _:
            raise ValueError(f"Unknown multi-score strategy: {multi_score_strategy}")


def score_from_runs(  # noqa: C901
    run_infos: list[dict[str, float]],
    scoring_metric: list[str] | str,
    multi_score_strategy: Literal["harmonic", "arithmetic", "geometric", "min"] = "harmonic",
) -> float:
    """Calculate a score from run metrics.

    Each metric can be provided by either ":higher" or ":lower" to indicate the direction of the metrics.
    This allows to correctly combine multiple metrics by doing 1/metric before calculation if a metric is ":lower".
    If no direction is provided, it is assumed to be ":higher".
    Has no real effect on the single score calculation, since only the mean is calculated there.

    In a multi-score setting, the score is calculated by combine-then-reduce the metrics.
    Meaning that first for each run the metrics are combined using the specified strategy,
    and then the results are reduced via mean.
    Please refer to the documentation to understand the different multi-score strategies.

    Ignores unstable runs when multi_score_strategy is "harmonic" or "geometric"
    If no runs are left, return 0.
    An unstable run is where one of the metrics is not finite or zero.

    Args:
        run_infos (list[dict[str, float]]): List of dictionaries containing run information and metrics
        scoring_metric (list[str] | str): Metric(s) to use for scoring.
        multi_score_strategy (Literal["harmonic", "arithmetic", "geometric", "min"], optional):
            Strategy for combining multiple metrics. Defaults to "harmonic".

    Returns:
        float: The calculated score

    Raises:
        ValueError: If an unknown multi-score strategy is provided.

    """
    # Single score in list
    if isinstance(scoring_metric, list) and len(scoring_metric) == 1:
        scoring_metric = scoring_metric[0]

    # Case single score
    if isinstance(scoring_metric, str):
        # In case the use set a specific direction
        scoring_metric = scoring_metric.removesuffix(":higher").removesuffix(":lower")
        metric_values = [run_info[scoring_metric] for run_info in run_infos]
        score = mean(metric_values)
    # Case multiple scores
    elif isinstance(scoring_metric, list):
        scores = []
        for run_info in run_infos:
            # Check if we can calculate a score
            is_unstable = check_score_is_unstable(run_info, scoring_metric)
            if is_unstable and multi_score_strategy in ["harmonic", "geometric"]:
                continue
            metric_values = []
            for metric in scoring_metric:
                higher_is_better = False if metric.endswith(":lower") else True
                metric = metric.removesuffix(":higher").removesuffix(":lower")
                val = run_info[metric]
                metric_values.append(val if higher_is_better else 1 / val)

            match multi_score_strategy:
                case "harmonic":
                    run_score = harmonic_mean(metric_values)
                case "arithmetic":
                    run_score = mean(metric_values)
                case "min":
                    run_score = min(metric_values)
                case "geometric":
                    run_score = geometric_mean(metric_values)
                case _:
                    raise ValueError("If an unknown multi-score strategy is provided.")
            scores.append(run_score)
        if len(scores) == 0:
            score = 0.0
        elif len(scores) == 1:
            score = scores[0]
        else:
            score = mean(scores)

    return score


def check_score_is_unstable(run_info: dict, scoring_metric: list[str] | str) -> bool:
    """Check the stability of the scoring metric.

    If any metric value is not finite or equal to zero, the scoring metric is considered unstable.

    Args:
        run_info (dict): The run information.
        scoring_metric (list[str] | str): The scoring metric.

    Returns:
        bool: True if the scoring metric is unstable, False otherwise.

    Raises:
        ValueError: If an unknown scoring metric type is provided.

    """
    # Single score in list
    if isinstance(scoring_metric, list) and len(scoring_metric) == 1:
        scoring_metric = scoring_metric[0]

    if isinstance(scoring_metric, str):
        metric_value = run_info[scoring_metric]
        is_unstable = not isfinite(metric_value) or metric_value == 0
        return is_unstable
    elif isinstance(scoring_metric, list):
        metric_values = [run_info[metric] for metric in scoring_metric]
        is_unstable = any(not isfinite(val) or val == 0 for val in metric_values)
        return is_unstable
    else:
        raise ValueError("Invalid scoring metric type")
