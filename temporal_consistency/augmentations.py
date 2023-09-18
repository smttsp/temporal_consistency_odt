import random
from albumentations import (
    Compose,
    RandomBrightnessContrast,
    RandomGamma,
    ColorJitter,
    ChannelShuffle,
    RGBShift,
    Blur,
    GaussNoise,
    RandomRain,
    RandomSnow,
    RandomSunFlare,
    RandomFog,
    RandomShadow,
    InvertImg,
    RandomToneCurve,
    Solarize,
    Equalize,
    Posterize,
)


def get_aug_list():
    return [
        RandomBrightnessContrast(p=random.uniform(0.5, 1.0)),
        RandomGamma(p=random.uniform(0.5, 1.0)),
        ColorJitter(
            brightness=random.uniform(0.1, 0.4),
            contrast=random.uniform(0.1, 0.5),
            saturation=random.uniform(0.1, 0.5),
            hue=random.uniform(0.1, 0.5),
            p=random.uniform(0.5, 1.0),
        ),
        ChannelShuffle(p=random.uniform(0.5, 1.0)),
        RGBShift(p=random.uniform(0.5, 1.0)),
        Blur(blur_limit=3, p=random.uniform(0.5, 1.0)),
        GaussNoise(p=random.uniform(0.5, 1.0)),
        RandomRain(p=random.uniform(0.5, 1.0)),
        RandomSnow(p=random.uniform(0.5, 1.0)),
        RandomSunFlare(p=random.uniform(0.5, 1.0), flare_roi=(0, 0, 1, 0.5)),
        RandomFog(p=random.uniform(0.5, 1.0)),
        RandomShadow(p=random.uniform(0.5, 1.0)),
        InvertImg(p=random.uniform(0.5, 1.0)),
        RandomToneCurve(p=random.uniform(0.5, 1.0)),
        Solarize(p=random.uniform(0.5, 1.0)),
        Equalize(p=random.uniform(0.5, 1.0)),
        Posterize(p=random.uniform(0.5, 1.0)),
    ]


def get_random_augmentation(image, num_aug=0):
    image_aug = image
    if num_aug > 0:
        augmentation_pipeline = Compose(
            random.sample(get_aug_list(), k=num_aug)
        )
        image_aug = augmentation_pipeline(image=image)["image"]
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(image_aug)
    # plt.show()

    return image_aug
