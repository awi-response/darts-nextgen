"""Create dataset and dataloader."""

import logging

import torch.utils.data

logger = logging.getLogger(__name__)


def create_dataloader(dataset, dataset_opt, phase):
    """Create dataloader."""
    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=dataset_opt["use_shuffle"],
            num_workers=dataset_opt["num_workers"],
            pin_memory=True,
        )
    elif phase == "val":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=dataset_opt["use_shuffle"],
            num_workers=16,
            pin_memory=True,
        )
    else:
        raise NotImplementedError(f"Dataloader [{phase:s}] is not found.")


def create_dataset(dataset_opt, phase):
    """Create dataset."""
    mode = dataset_opt["mode"]
    from darts_superresolution.data_processing.LRHR_dataset import LRHRDataset

    # print("before creation: ", phase, dataset_opt['data_len'])
    dataset = LRHRDataset(
        dataroot=dataset_opt["dataroot"],
        datatype=dataset_opt["datatype"],
        l_resolution=dataset_opt["l_resolution"],
        r_resolution=dataset_opt["r_resolution"],
        split=phase,
        data_len=dataset_opt["data_len"],
        need_LR=(mode == "LRHR"),
    )
    # print("After creation: ", phase, dataset_opt['data_len'])
    logger.info("Dataset [{:s} - {:s}] is created.".format(dataset.__class__.__name__, dataset_opt["name"]))
    return dataset
