"""Weather and photometric augmentation for GV-YOLO26 (applied on top of the standard Ultralytics recipe)."""


def weather_augmentations():
    import albumentations as A

    return [
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.20),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.25),
        A.Sharpen(alpha=(0.2, 0.4), p=0.15),
        A.RandomRain(drop_length=15, blur_value=3, brightness_coefficient=0.85, p=0.10),
        A.RandomSnow(snow_point_range=(0.1, 0.25), brightness_coeff=1.8, p=0.05),
        A.MotionBlur(blur_limit=(3, 9), p=0.10),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0, p=0.10),
    ]
