"""Small visuals previews for the output."""

import matplotlib.pyplot as plt
import xarray as xr


def thumbnail(tile: xr.Dataset) -> plt.Figure:
    """Create a thumbnail of the tile.

    Args:
        tile (xr.Dataset): The tile to create a thumbnail from.

    Returns:
        plt.Figure: The figure with the thumbnail.

    """
    prev_res = 512  # Prefered resolution for the thumbnail, will not exactly match
    orig_res = max(tile.sizes.values())
    if orig_res > prev_res:
        factor = int(orig_res / prev_res)
        tile = tile.odc.reproject(tile.odc.geobox.zoom_out(factor))

    tile_id = tile.attrs.get("s2_id", "unknown")

    # Add some statistics
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f"Tile {tile_id} (lowres) [epsg:{tile.odc.crs.epsg}]")
    rgba = tile.odc.to_rgba(bands=["red", "green", "blue"], vmin=0, vmax=2000)
    rgba.plot.imshow(ax=ax)

    # Prediction boundaries
    tile.probabilities.plot.contour(ax=ax, levels=[0.5])
    # Validity mask
    tile.extent.plot.contour(ax=ax, levels=[0.5], colors="r", alpha=0.5)
    return fig
