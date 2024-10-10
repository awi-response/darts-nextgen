# Logging

We want to use the python logging module as much as possible to traceback errors and document the pipeline processes.
Furthermore, we want to configure each logger with the `RichHandler`, which prettyfies the output with [rich](https://github.com/Textualize/rich).

## Usage Guide

For logging inside a package should be done without any further configuration:

```py
import logging

logger = logging.getLogger(__name__) # don't replace __name__
```

Logging at a top-level can and should be further configured:

> Code is untested!

```py
import logging

from rich.logging import RichHandler

console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logging.basicConfig(handlers=[console_handler, file_handler])
```

### Supressing Arrays

When printing or logging large numpy arrays a lot of numbers get truncated, however the array still takes a lot of space. Using `lovely_numpy` and `lovely_tensor` can help here:

```py
import numyp as np
import torch
import xarray as xr
from lovely_numpy import lo
from lovely_tensors import monkey_patch

monkey_patch()
xr.set_options(display_expand_data=False)

a = np.zeros((8, 1024, 1024))
la = lo(a)
da = xr.DataArray(a)
t = torch.tensor(a)

logger.warning(la)
logger.warning(da)
logger.warning(t)
```
