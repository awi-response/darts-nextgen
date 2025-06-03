# Debugging

!!! Tip "CUDA related"

    There is a complete [page about CUDA related errors](../dev/cuda_fixes.md).

To produce more (hopefully) helpful output from the CLI, two flags can be set:

- `--verbose` This will set the log-level to DEBUG. The output becomes much more noisy, but contains a lot of useful information.
- `--tracebacks-show-locals` This will enable the rich traceback feature to not only show the traceback when errors are occuring, but also the local variables in that scope. This, however, is much slower and can potentially output to much information for the terminal to handle.
