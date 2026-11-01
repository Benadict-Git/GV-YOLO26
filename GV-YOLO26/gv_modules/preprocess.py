"""Road surface enhancement for unpainted/unmarked asphalt and concrete speed bumps.

Unmarked speed bumps have near-zero color contrast against the surrounding pavement.
Detection relies on subtle surface relief, geometric shadow gradients, and road surface deformation.
This module enhances local luminance contrast and pavement relief without oversaturating headlights or sky.
"""

from typing import Optional
import cv2
import numpy as np


def enhance_road_surface(
    img_bgr: np.ndarray,
    clip_limit: float = 2.2,
    tile_grid_size: int = 8,
    unsharp_strength: float = 0.35,
) -> np.ndarray:
    """Amplify subtle surface relief, shadow gradients, and asphalt seams of unmarked speed bumps.

    Operates in LAB color space:
    1. Isolates the L (Luminance) channel to avoid chrominance distortions.
    2. Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the road plane.
    3. Blends a subtle unsharp mask to accentuate 3D bump crest edges and elevation transitions.

    Args:
        img_bgr: Input BGR image (numpy array, uint8).
        clip_limit: Contrast threshold for CLAHE.
        tile_grid_size: Size of local equalization grid (default 8x8).
        unsharp_strength: Weight for high-frequency edge accentuation (0.0 to 1.0).

    Returns:
        Enhanced BGR image with amplified road relief details.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    # Convert to LAB to decouple luminance from color channels
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 1. CLAHE on Luminance
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_clahe = clahe.apply(l_channel)

    # 2. Subtle unsharp mask to bring out pavement crest gradients
    if unsharp_strength > 0.0:
        blurred = cv2.GaussianBlur(l_clahe, (0, 0), sigmaX=3)
        # Unsharp: original + strength * (original - blurred)
        l_sharpened = cv2.addWeighted(l_clahe, 1.0 + unsharp_strength, blurred, -unsharp_strength, 0)
    else:
        l_sharpened = l_clahe

    # Merge channels back and return BGR
    merged_lab = cv2.merge((l_sharpened, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
