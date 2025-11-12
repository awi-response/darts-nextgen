---
hide:
  - navigation
---

# CLI Reference

The CLI for DARTS is built using the [Cyclopts](https://cyclopts.readthedocs.io/en/latest/) library, which provides a flexible framework for creating command-line applications in Python.
Cyclopts allows for easy generation of a CLI based on existing functions and their type annotations and docstrings.

The commands of the DARTS CLI are very nested.
Some flags are global these are documented in the `darts --help`, but not in the help of subcommands.

## Top-Level Commands

::: cyclopts
    module: darts.cli:app
    exclude-commands: [training, inference]
    generate-toc: false
    flatten-commands: true
    heading-level: 3

## Training Commands

::: cyclopts
    module: darts.cli:app
    commands: [training]
    exclude-commands: [training.create_dataset]
    generate-toc: false
    flatten-commands: true
    heading-level: 3

## Create Dataset Commands

::: cyclopts
    module: darts.cli:app
    commands: [training.create_dataset]
    generate-toc: false
    flatten-commands: true
    heading-level: 3

## Inference Commands

::: cyclopts
    module: darts.cli:app
    commands: [inference]
    exclude-commands: [inference.prep-data]
    generate-toc: false
    flatten-commands: true
    heading-level: 3

## Prepare Data Commands

::: cyclopts
    module: darts.cli:app
    commands: [inference.prep-data]
    generate-toc: false
    flatten-commands: true
    heading-level: 3
