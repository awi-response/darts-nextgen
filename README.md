# DARTS nextgen

[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye.astral.sh)
[![Lint](https://github.com/awi-response/darts-nextgen/actions/workflows/ruff.yml/badge.svg)](https://github.com/awi-response/darts-nextgen/actions/workflows/ruff.yml)
[![CI](https://github.com/awi-response/darts-nextgen/actions/workflows/update_version.yml/badge.svg)](https://github.com/awi-response/darts-nextgen/actions/workflows/update_version.yml)

> Early Alpha!

Panarctic Database of Active Layer Detachment Slides and Retrogressive Thaw Slumps from Deep Learning on High Resolution Satellite Imagery.
This is te successor of the thaw-slump-segmentation (pipeline), with which the first version of the DARTS dataset was created.

## Editor setup

There is only setup files provided for VSCode and no other editor (yet).
A list of extensions and some settings can be found in the `.vscode`.
At the first start, VSCode should ask you if you want to install the recommended extension.
The settings should be automaticly used by VSCode.
Both should provide the developers with a better experience and enforce code-style.

## Environment setup

Prereq:

- [Rye](https://rye.astral.sh/): `curl -sSf https://rye.astral.sh/get | bash`
- [GDAL](https://gdal.org/en/latest/index.html): `sudo apt update && sudo apt install libpq-dev gdal-bin libgdal-dev` or for HPC `conda install conda-forge::gdal`
- Clang: `sudo apt update && sudo apt install clang` or for HPC `conda install conda-forge::clang_linux-64`

> If you install GDAL via apt for linux you can view the supported versions here: <https://pkgs.org/search/?q=libgdal-dev>. For a finer controll over the versions please use conda.

Now first check your gdal-version:

```sh
$ gdal-config --version
3.9.2
```

And your CUDA version (if you want to use CUDA):

```sh
$ nvidia-smi
# Now look on the top right of the table
```

> The GDAL version is relevant, since the version of the python bindings needs to match the installed GDAL version

Now, to sync with a specific `gdal` version, add `gdalXX` to the `--features` flag.
To sync with a specific `cuda` version, add `cuda1X` or without cuda `cpu`.
E.g.:

```sh
rye sync -f --features gdal39,cuda12 # For CUDA 12 and GDAL 3.9.2
```

As of right now, the supported `gdal` versions are: 3.9.2 (`gdal39`), 3.8.5 (`gdal38`), 3.8.4 (`gdal384`), 3.7.3 (`gdal37`) and 3.6.4 (`gdal36`).
If your GDAL version is not supported (yet) please sync without GDAL and then install GDAL to an new optional group. For example, if your GDAL version is 3.8.4:

```sh
rye sync -f
rye add --optional=gdal384 "gdal==3.8.4"
```

> IMPORTANT! If you installed any of clang or gdal with conda, please ensure that while installing dependencies and working on the project to have the conda environment activated in which you installed clang and or gdal.

### Troubleshoot: Rye can't find the right versions

Because the `pyproject.toml` specifies additional sources, e.g. `https://download.pytorch.org/whl/cpu`, it can happen that the a package with an older version is found in these package-indexes.
If such a version is found, `uv` (the installer behind `Rye`) currently stops searching other sources for the right version and stops with an `Version not found` error.
This can look something like this:

```sh
No solution found when resolving dependencies:
  ╰─▶ Because only torchmetrics==1.0.3 is available and you require torchmetrics>=1.4.1, we can conclude that your requirements are unsatisfiable.
```

To fix this you can set an environment variable to tell `uv` to search all package-indicies:

```sh
UV_INDEX_STRATEGY="unsafe-best-match" rye sync ...
```

I recommend adding the following to your `.zshrc` or `.bashrc`:

```sh
# Change the behaviour of uv package resolution to enable additional sources without breaking existing version-requirements
export UV_INDEX_STRATEGY="unsafe-best-match"
```

Please see these issues:

- [Rye: Can't specify per-dependency package index / can't specify uv behavior in config file](https://github.com/astral-sh/rye/issues/1210#issuecomment-2263761535)
- [UV: Add support for pinning a package to a specific index](https://github.com/astral-sh/uv/issues/171)
