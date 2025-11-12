# Training (Binary Segmentation)

To train a simple SMP (Segmentation Model Pytorch) model you can use the command:

```sh
[uv run] darts training train-smp --your-args-here ...
```

!!! tip "Model Architecture"
    Configurations for the architecture and encoder can be found in the [SMP documentation](https://smp.readthedocs.io/en/latest/index.html) for model configurations.

!!! warning "Change defaults"
    Even though the defaults from the CLI are somewhat useful, it is recommended to create a config file and change the behavior of the training there.

This command will train a simple SMP model on the data in the `train-data-dir` directory.
The training relies on PyTorch Lightning, which is a high-level interface for PyTorch.
It is recommended to use Weights and Biases (wandb) for the logging, because the training script is heavily influenced by how the organization of wandb works.

The training follows the data splitting, decribed in the [Data Guide](./data.md) and [Cross-Validation Guide](./cv.md)
To test the model on the test split, you can use the following command:

```sh
[uv run] darts training test-smp --your-args-here ...
```

??? tip "You can also use the underlying functions directly:"

    [darts_segmentation.training.train_smp][]
    [darts_segmentation.training.test_smp][]

## Data splits

The initial training/test data split is performed at train/test time by using the `data_split_method` and `data_split_by` parameters.
`data_split_method` can be one of the following:

- `"random"` will split the data randomly, the seed is always 42 and the size of the test set can be specified by providing a list with a single float between 0 and 1 to `data_split_by`.
- `"region"` will split the data by one or multiple regions, which can be specified by providing a str or list of str to `data_split_by`.
- `"sample"` will split the data by sample ids, which can be specified similar to `"region"`.
- `None`, no split is done and the complete dataset is used for both training and testing.
