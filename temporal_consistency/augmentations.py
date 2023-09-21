"""This module provides functions to generate a list of image augmentations
using the albumentations library and to apply a random subset of those
augmentations to a given image. Depending on the `num_aug` input, the
`get_random_augmentation` function selects a specified number of augmentations
 from this list of possible options and applies them to an input image.
"""

import random

from albumentations import (
    Blur,
    ChannelShuffle,
    ColorJitter,
    Compose,
    Equalize,
    GaussNoise,
    InvertImg,
    Posterize,
    RandomBrightnessContrast,
    RandomFog,
    RandomGamma,
    RandomRain,
    RandomShadow,
    RandomSnow,
    RandomSunFlare,
    RandomToneCurve,
    RGBShift,
    Solarize,
)


def get_aug_list():
    """Returns a list of augmentations to be applied to the frames."""

    return [
        RandomBrightnessContrast(p=1.0),
        RandomGamma(p=1.0),
        ColorJitter(
            brightness=random.uniform(0.1, 0.4),
            contrast=random.uniform(0.1, 0.5),
            saturation=random.uniform(0.1, 0.5),
            hue=random.uniform(0.1, 0.5),
            p=1.0,
        ),
        ChannelShuffle(p=1.0),
        RGBShift(p=1.0),
        Blur(blur_limit=3, p=1.0),
        GaussNoise(p=1.0),
        RandomRain(p=1.0),
        RandomSnow(p=1.0),
        RandomSunFlare(p=1.0, flare_roi=(0, 0, 1, 0.5)),
        RandomFog(p=1.0),
        RandomShadow(p=1.0),
        InvertImg(p=1.0),
        RandomToneCurve(p=1.0),
        Solarize(p=1.0),
        Equalize(p=1.0),
        Posterize(p=1.0),
    ]


def get_random_augmentation(image, num_aug=0):
    """Pick random augmentations from the list and apply them to the image.
    The number of augmentations to apply is specified by num_aug.
    """

    image_aug = image
    if num_aug > 0:
        augmentation_pipeline = Compose(
            random.sample(get_aug_list(), k=num_aug)
        )
        image_aug = augmentation_pipeline(image=image)["image"]
    return image_aug
