# Logging Guide

We want to use the python logging module as much as possible to traceback errors and document the pipeline processes.
Furthermore, we want to configure each logger with the `RichHandler`, which prettyfies the output with [rich](https://github.com/Textualize/rich).

## Setup Guide

Currently, all setup related to logging is found in the `darts.utils.logging.py` file.
It contains two functions:

1. A setup function which sets the log-level for all `darts.*` logger and add default options to xarray and pytorch to supress arrays. See how to [supress arrays](#supressing-arrays).
2. A function which adds a file and a rich log handler.

Both functions are used in the CLI setup but can also be called from e.g. a notebook. The recommended approach for handling logging within a notebook is the following:

```python
import logging
from rich.logging import RichHandler
from darts.utils.logging import LoggingManager

LoggingManager.setup_logging(verbose=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
```

This way the notebook won't spam logfiles everywhere and we still have control over our rich handler.

## Usage Guide

For logging inside a darts-package should be done without any further configuration:

```py
import logging

logger = logging.getLogger(__name__.replace("darts_", "darts.")) # don't replace __name__
```

Logging at the top-level `darts` package can just use a `__name__` logger:

```py
import loggin

logger = logging.getLogger(__name__) # don't replace __name__
```

### Supressing Arrays

When printing or logging large numpy arrays a lot of numbers get truncated, however the array still takes a lot of space.
Using `lovely_numpy` and `lovely_tensor` can help here:

```py
import numyp as np
import torch
import xarray as xr
from lovely_numpy import lo
import lovely_tensors

lovely_tensors.monkey_patch()
lovely_tensors.set_config(color=False)  # Disable colored output, which is better for logging
xr.set_options(display_expand_data=False)

a = np.zeros((8, 1024, 1024))
la = lo(a)
da = xr.DataArray(a)
t = torch.tensor(a)

logger.warning(la)
logger.warning(da)
logger.warning(t)
```

## Dev Guide

This section should cover best practices developing inside the darts package.

- Fail fast
- Logging exceptions to debug

### Exceptions

When logging exceptions from within an component, the complete exception should be logged with the traceback by setting `exc_info=True` to debug and then re-raised:

```python
try:
    # some code
except Exception as e:
    logger.debug(e, exc_info=True)
    raise
```
