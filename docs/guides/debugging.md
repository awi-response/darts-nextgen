# Debugging

!!! Tip "CUDA related"

    There is a complete [page about CUDA related errors](../dev/cuda_fixes.md).

To produce more (hopefully) helpful output from the CLI, the verbosity level can be set via the verbosity flags:

- `-v` This will set the log-level of the all DARTS modules to DEBUG. The output becomes much more noisy, but contains a lot of useful information.
- `-vv` This will do the same as `-v`, but in addition also show local in tracebacks via `rich.tracebacks.install(show_locals=True)` and also set the log-level of some third-party libraries (like PyTorch) to INFO as well. This, however, is much slower and can potentially output to much information for the terminal to handle.
- `-vvv` This will do the same as `-vv`, but instead of setting third-party libraries to INFO, it will set them to DEBUG. This is mainly useful for debugging issues with third-party libraries.

By using the `--log-plain` flag the loggers will not use `rich` to print to the console, this should help working on machines with poor / no TTY support, like SLURM applications.
