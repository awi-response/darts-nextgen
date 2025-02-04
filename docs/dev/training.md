# Training

!!! info "Incomplete docs"
    This page is still under construction.

## Simple SMP train and test

To train a simple SMP (Segmentation Model Pytorch) model, you first need to preprocess your S2 data:

```sh
[rye run] darts preprocess-s2-train-data --your-args-here ... 
```

This will create three data splits:

- `cross-val`, used for train and validation
- `val-test` 5% random leave-out for testing the randomness distribution shift of the data
- `test` leave-out region for testing the spatial distribution shift of the data

Now you can train a model:

```sh
[rye run] darts train-smp --your-args-here ...
```

!!! warning "Change defaults"
    Even though the defaults from the CLI are somewhat useful, it is recommended to create a config file and change the behavior of the training there.

This will train a model with the `cross-val` data and save the model to disk.
The training relies on PyTorch Lightning, which is a high-level interface for PyTorch.
You can now test the model on the other two splits:

```sh
[rye run] darts test-smp --your-args-here ...
```

The checkpoint stored is not usable for the pipeline yet, since it is stored in a different format.
To convert the model to a format, you need to convert is first:

```sh
[rye run] darts convert-lightning-checkpoint --your-args-here ...
```
