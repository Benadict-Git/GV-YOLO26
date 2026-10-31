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


def dusk_augmentations():
    """Augmentations targeting dusk, low-contrast, shadow, and faded marking edge cases."""
    import albumentations as A

    return [
        # Simulates dusk / shadows on asphalt (tree silhouettes, overpasses)
        A.RandomShadow(num_shadows_limit=(1, 3), shadow_intensity_range=(0.4, 0.7), p=0.40),
        # Severe low-light and underexposed road surface
        A.RandomBrightnessContrast(brightness_limit=(-0.60, -0.20), contrast_limit=(-0.25, 0.25), p=0.45),
        # High-contrast equalization for faint/faded speed bump markings
        A.CLAHE(clip_limit=8.0, tile_grid_size=(8, 8), p=0.40),
        # Breaks photometric texture reliance so dark asphalt is not misclassified as pothole
        A.Solarize(threshold_range=(0.25, 0.60), p=0.20),
        # Dusk tone compression across color channels
        A.RandomToneCurve(scale=0.35, p=0.30),
        # Motion blur from vehicle movement in dim lighting
        A.MotionBlur(blur_limit=(3, 7), p=0.15),
    ]


def combined_edge_augmentations():
    """Combines weather and dusk/low-contrast pipelines for robust edge-case coverage."""
    return weather_augmentations() + dusk_augmentations()
