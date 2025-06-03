# ruff: noqa: D100, D103
# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

iou = np.arange(0, 1, 0.01)
iou = xr.DataArray(iou, dims=["iou"], coords={"iou": iou})
loss = np.arange(0, 2, 0.01)
loss = 1 / xr.DataArray(loss, dims=["loss"], coords={"loss": loss})
recall = np.arange(0, 1, 0.01)
recall = xr.DataArray(recall, dims=["recall"], coords={"recall": recall})
# %%


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


# %%
viz_strategies(iou, loss).savefig("../docs/assets/score_strategies_iou_loss.png")
viz_strategies(iou, recall).savefig("../docs/assets/score_strategies_iou_recall.png")
