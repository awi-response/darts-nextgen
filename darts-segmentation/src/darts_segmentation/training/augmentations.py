"""Augmentations for segmentation tasks."""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import albumentations as A  # noqa: N812

""" Implementation of augmentations for segmentation tasks."""
Augmentation = Literal[
    "HorizontalFlip",
    "VerticalFlip",
    "RandomRotate90",
    "Blur",
    "RandomBrightnessContrast",
    "MultiplicativeNoise",
]


def get_augmentation(augment: list[Augmentation] | None) -> "A.Compose | None":
    """Get augmentations for segmentation tasks.

    Args:
        augment (list[Augmentation] | None): List of augmentations to apply.
            If None or emtpy, no augmentations are applied.
            If not empty, augmentations are applied in the order they are listed.
            Available augmentations:
                - HorizontalFlip
                - VerticalFlip
                - RandomRotate90
                - Blur
                - RandomBrightnessContrast
                - MultiplicativeNoise

    Raises:
        ValueError: If an unknown augmentation is provided.

    Returns:
        A.Compose | None: A Compose object containing the augmentations.
            If no augmentations are provided, returns None.

    """
    import albumentations as A  # noqa: N812

    if not isinstance(augment, list) or len(augment) == 0:
        return None
    transforms = []
    for aug in augment:
        match aug:
            case "HorizontalFlip":
                transforms.append(A.HorizontalFlip())
            case "VerticalFlip":
                transforms.append(A.VerticalFlip())
            case "RandomRotate90":
                transforms.append(A.RandomRotate90())
            case "Blur":
                transforms.append(A.Blur())
            case "RandomBrightnessContrast":
                transforms.append(A.RandomBrightnessContrast())
            case "MultiplicativeNoise":
                transforms.append(A.MultiplicativeNoise(per_channel=True, elementwise=True))
            case _:
                raise ValueError(f"Unknown augmentation: {aug}")
    return A.Compose(transforms)
