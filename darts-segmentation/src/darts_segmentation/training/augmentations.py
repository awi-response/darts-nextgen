"""Augmentations for segmentation tasks."""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import albumentations as A  # noqa: N812

""" Implementation of augmentations for segmentation tasks."""
Augmentation = Literal[
    "HorizontalFlip",
    "VerticalFlip",
    "RandomRotate90",
    "D4",
    "Blur",
    "RandomBrightnessContrast",
    "MultiplicativeNoise",
    "Posterize",
]


def get_augmentation(augment: list[Augmentation] | None, always_apply: bool = False) -> "A.Compose | None":  # noqa: C901
    """Get augmentations for segmentation tasks.

    Args:
        augment (list[Augmentation] | None): List of augmentations to apply.
            If None or emtpy, no augmentations are applied.
            If not empty, augmentations are applied in the order they are listed.
            Available augmentations:
                - D4 (Combination of HorizontalFlip, VerticalFlip, and RandomRotate90)
                - Blur
                - RandomBrightnessContrast
                - MultiplicativeNoise
                - Posterize (quantization to reduce number of bits per channel)
        always_apply (bool): If True, augmentations are always applied.
            This is useful for visualization/testing augmentations.
            Default is False.

    Raises:
        ValueError: If an unknown augmentation is provided.

    Returns:
        A.Compose | None: A Compose object containing the augmentations.
            If no augmentations are provided, returns None.

    """
    import albumentations as A  # noqa: N812

    if not isinstance(augment, list) or len(augment) == 0:
        return None

    # Replace HorizontalFlip, VerticalFlip, RandomRotate90 with D4
    if "HorizontalFlip" in augment and "VerticalFlip" in augment and "RandomRotate90" in augment:
        augment = [aug for aug in augment if aug not in ("HorizontalFlip", "VerticalFlip", "RandomRotate90")]
        augment.insert(0, "D4")

    transforms = []
    for aug in augment:
        match aug:
            case "D4":
                transforms.append(A.D4())
            case "HorizontalFlip":
                transforms.append(A.HorizontalFlip(p=1.0 if always_apply else 0.5))
            case "VerticalFlip":
                transforms.append(A.VerticalFlip(p=1.0 if always_apply else 0.5))
            case "RandomRotate90":
                transforms.append(A.RandomRotate90())
            case "Blur":
                transforms.append(A.Blur(p=1.0 if always_apply else 0.5))
            case "RandomBrightnessContrast":
                transforms.append(A.RandomBrightnessContrast(p=1.0 if always_apply else 0.5))
            case "MultiplicativeNoise":
                transforms.append(
                    A.MultiplicativeNoise(per_channel=True, elementwise=True, p=1.0 if always_apply else 0.5)
                )
            case "Posterize":
                # First convert to uint8, then apply posterization, then convert back to float32
                # * Note: This does only work for float32 images.
                transforms += [
                    A.FromFloat(dtype="uint8"),
                    A.Posterize(num_bits=6, p=1.0),
                    A.ToFloat(),
                ]
            case _:
                raise ValueError(f"Unknown augmentation: {aug}")
    return A.Compose(transforms)
