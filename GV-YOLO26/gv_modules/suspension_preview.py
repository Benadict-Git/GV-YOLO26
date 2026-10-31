"""Active suspension preview control: monocular distance, Time-to-Impact (TTI) and damping commands.

Computes road-plane longitudinal distance using pinhole geometry:
    alpha = arctan((v_bottom - v0) / f_v)
    theta = pitch + alpha
    Z = cam_height / tan(theta)
and calculates Time-to-Impact (TTI) and proactive damping control commands.
"""

import math
from typing import Dict, Optional, Tuple


class SuspensionPreview:
    """Pinhole ground-plane distance and Time-to-Impact estimator for active vehicle suspension."""

    def __init__(
        self,
        cam_height_m: float = 1.25,
        pitch_deg: float = 8.0,
        vfov_deg: float = 50.0,
        default_ego_speed_kmh: float = 60.0,
        min_dist_m: float = 2.0,
        max_dist_m: float = 45.0,
    ):
        self.cam_height_m = cam_height_m
        self.pitch_rad = math.radians(pitch_deg)
        self.vfov_rad = math.radians(vfov_deg)
        self.default_ego_speed_kmh = default_ego_speed_kmh
        self.min_dist_m = min_dist_m
        self.max_dist_m = max_dist_m

    def estimate_distance(
        self,
        y2: float,
        img_h: int,
        y1: Optional[float] = None,
    ) -> float:
        """Estimate longitudinal ground distance Z (in meters) to hazard from bottom pixel coordinate y2."""
        v0 = img_h / 2.0
        fv = (img_h / 2.0) / math.tan(self.vfov_rad / 2.0)

        # Pixel offset from optical center to contact patch
        delta_v = max(1.0, y2 - v0)
        alpha = math.atan(delta_v / fv)
        total_angle = self.pitch_rad + alpha

        if total_angle <= 0.01:
            return self.max_dist_m

        z_est = self.cam_height_m / math.tan(total_angle)
        return float(min(self.max_dist_m, max(self.min_dist_m, z_est)))

    def compute_preview(
        self,
        box_xyxy: Tuple[float, float, float, float],
        img_h: int,
        hazard_class: str,
        ego_speed_kmh: Optional[float] = None,
    ) -> Dict[str, any]:
        """Compute distance, TTI, and suspension preview damping command for a detected hazard."""
        speed_kmh = ego_speed_kmh if ego_speed_kmh is not None else self.default_ego_speed_kmh
        speed_ms = max(1.0, speed_kmh / 3.6)

        x1, y1, x2, y2 = box_xyxy
        z_m = self.estimate_distance(y2, img_h, y1)
        tti_ms = (z_m / speed_ms) * 1000.0

        # Active Suspension Damping Strategy
        # Preview zone: 15.0m to 30.0m -> proactive softening to prevent impact spike
        # Critical arrival zone: < 15.0m -> pre-load damper valving / active anti-dive
        is_hazard = hazard_class in ("Speed Bump", "Pothole", "speed_bump", "pothole")
        if is_hazard:
            if z_m < 15.0:
                damping_cmd = "PRELOAD_CRITICAL"
                damping_color = (0, 0, 255)  # Red alert
            elif 15.0 <= z_m <= 30.0:
                damping_cmd = "SOFTEN_FAST"
                damping_color = (0, 165, 255)  # Orange warning
            else:
                damping_cmd = "PREVIEW_TRACK"
                damping_color = (0, 255, 255)  # Yellow tracking
        else:
            damping_cmd = "NOMINAL"
            damping_color = (50, 220, 90)  # Green

        return {
            "z_m": round(z_m, 1),
            "tti_ms": int(tti_ms),
            "damping_cmd": damping_cmd,
            "damping_color": damping_color,
            "ego_speed_kmh": round(speed_kmh, 1),
        }
