"""Image segmentation of thaw-slumps for the DARTS dataset."""

import importlib.metadata

# TODO: We don't re-export anything here to avoid heavy imports when importing train functions for the CLI.
# Find a better solution than just don't re-export anything.

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
