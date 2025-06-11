"""Benchmark related utilities."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def benchviz(
    stopuhr_data: Path,
    *,
    viz_dir: Path | None = None,
):
    """Visulize benchmark based on a Stopuhr data file produced by a pipeline run.

    !!! note
        This function changes the seaborn theme to "whitegrid" for better visualization.

    Args:
        stopuhr_data (Path): Path to the Stopuhr data file.
        viz_dir (Path | None): Path to the directory where the visualization will be saved.
            If None, the defaults to the parent directory of the Stopuhr data file.
            Defaults to None.

    Returns:
        plt.Figure: A matplotlib figure containing the benchmark visualization.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # Visualize the results
    sns.set_theme(style="whitegrid")

    assert stopuhr_data.suffix == ".parquet", "Stopuhr data file must be a parquet file."

    times = pd.read_parquet(stopuhr_data)
    times_long = times.melt(ignore_index=False, value_name="time", var_name="step").reset_index(drop=False)
    times_desc = times.describe()
    times_sum = times.sum()

    # Pretty print the results
    for col in times_desc.columns:
        mean = times_desc[col]["mean"]
        std = times_desc[col]["std"]
        total = times_sum[col]
        n = int(times_desc[col]["count"].item())
        logger.info(f"{col} took {mean:.2f} ± {std:.2f}s ({n=} -> {total=:.2f}s)")

    # axs: hist, histlog, bar, heat
    fig, axs = plt.subplot_mosaic(
        [
            ["histlog"] * 4,
            ["histlog"] * 4,
            ["hist", "hist", "heat", "heat"],
            ["hist", "hist", "heat", "heat"],
            ["bar", "bar", "bar", "bar"],
        ],
        layout="constrained",
        figsize=(20, 15),
    )

    sns.histplot(
        data=times_long,
        x="time",
        hue="step",
        bins=100,
        # log_scale=True,
        ax=axs["hist"],
    )
    axs["hist"].set_xlabel("Time in seconds")
    axs["hist"].set_title("Histogram of time taken for each step", fontdict={"fontweight": "bold"})

    sns.histplot(
        data=times_long,
        x="time",
        hue="step",
        bins=100,
        log_scale=True,
        kde=True,
        ax=axs["histlog"],
    )
    axs["histlog"].set_xlabel("Time in seconds")
    axs["histlog"].set_title("Histogram of time taken for each step (log scale)", fontdict={"fontweight": "bold"})

    sns.heatmap(
        times.T,
        robust=True,
        cbar_kws={"label": "Time in seconds"},
        ax=axs["heat"],
    )
    axs["heat"].set_xlabel("Sample")
    axs["heat"].set_title("Heatmap of time taken for each step and sample", fontdict={"fontweight": "bold"})

    bottom = np.array([0.0])
    for i, (step, time_taken) in enumerate(times.mean().items()):
        axs["bar"].barh(["Time taken"], [time_taken], label=step, color=sns.color_palette()[i], left=bottom)
        # Add a text label to the bar
        axs["bar"].text(
            bottom[-1] + time_taken / 2,
            0,
            f"{step}:\n{time_taken:.1f} s",
            va="center",
            ha="center",
            fontsize=10,
            color="white",
        )
        bottom += time_taken
    axs["bar"].legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3)
    # Make the y-axis labels vertical
    axs["bar"].set_yticks([0.15], labels=["Time taken"], rotation=90)
    axs["bar"].set_xlabel("Time in seconds")
    axs["bar"].set_title("Avg. time taken for each step", fontdict={"fontweight": "bold"})

    # Save the figure
    viz_dir = viz_dir or stopuhr_data.parent
    viz_dir.mkdir(parents=True, exist_ok=True)
    fpath = viz_dir / stopuhr_data.name.replace(".parquet", ".png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    logger.info(f"Benchmark visualization saved to {fpath.resolve()}")

    return fig
