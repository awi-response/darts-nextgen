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
    orgi_res = max(tile.sizes.values())
    factor = int(orgi_res / prev_res)
    tile_lowres = tile.odc.reproject(tile.odc.geobox.zoom_out(factor))

    tile_id = tile.attrs.get("s2_id", "unknown")

    # Add some statistics
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f"Tile {tile_id} (lowres) [epsg:{tile.odc.crs.epsg}]")
    rgba = tile_lowres.odc.to_rgba(bands=["red", "green", "blue"], vmin=0, vmax=0.2)
    rgba.plot.imshow(ax=ax)

    # Prediction boundaries
    tile_lowres.probabilities.where(tile_lowres.probabilities != 255).plot.contour(ax=ax, levels=[50])
    # Validitity mask
    (tile_lowres.probabilities == 255).plot.contour(ax=ax, levels=[0.5], colors="r", alpha=0.5)
    return fig
