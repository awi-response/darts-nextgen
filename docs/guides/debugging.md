# Debugging

!!! Tip "CUDA related"

    There is a complete [page about CUDA related errors](../dev/cuda_fixes.md).

To produce more (hopefully) helpful output from the CLI, two flags can be set:

- `--verbose` This will set the log-level of the all DARTS modules to DEBUG. The output becomes much more noisy, but contains a lot of useful information.
- `--very-verbose` This will do the same as `--verbose`, but in addition also show local in tracebacks and also set the log-level of some third-party libraries (like PyTorch) to INFO as well. This, however, is much slower and can potentially output to much information for the terminal to handle.
- `--debug` This will do the same as `--very-verbose`, but instead of setting third-party libraries to INFO, it will set them to DEBUG. This is mainly useful for debugging issues with third-party libraries.
