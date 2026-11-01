"""Geodesic hazard mapper: converts camera bounding box & distance into real-world GPS coordinates and Google Maps links."""

import math
from typing import Dict, Optional, Tuple


# WGS-84 Earth Mean Radius in meters
EARTH_RADIUS_M = 6371008.8


def project_hazard_gps(
    ego_lat: float,
    ego_lon: float,
    heading_deg: float,
    dist_longitudinal_m: float,
    dist_lateral_m: float = 0.0,
) -> Tuple[float, float]:
    """Project vehicle ground distance Z and lateral offset X into real-world GPS coordinates (WGS-84).

    Args:
        ego_lat: Ego vehicle latitude in decimal degrees.
        ego_lon: Ego vehicle longitude in decimal degrees.
        heading_deg: Ego vehicle heading angle in degrees (0 = North, 90 = East, 180 = South, 270 = West).
        dist_longitudinal_m: Distance Z ahead along vehicle axis in meters.
        dist_lateral_m: Lateral distance X perpendicular to vehicle axis in meters (positive = right, negative = left).

    Returns:
        (hazard_lat, hazard_lon) in decimal degrees rounded to 6 decimal places (~0.1m precision).
    """
    total_dist = math.hypot(dist_longitudinal_m, dist_lateral_m)
    if total_dist < 0.01:
        return round(ego_lat, 6), round(ego_lon, 6)

    # Relative angle from heading vector to hazard
    rel_angle_rad = math.atan2(dist_lateral_m, max(0.1, dist_longitudinal_m))
    true_bearing_rad = (math.radians(heading_deg) + rel_angle_rad) % (2.0 * math.pi)

    delta = total_dist / EARTH_RADIUS_M
    phi1 = math.radians(ego_lat)
    lambda1 = math.radians(ego_lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(true_bearing_rad)
    )

    lambda2 = lambda1 + math.atan2(
        math.sin(true_bearing_rad) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )

    # Normalize longitude to -180 to +180
    lambda2_deg = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    phi2_deg = math.degrees(phi2)

    return round(phi2_deg, 6), round(lambda2_deg, 6)


def estimate_lateral_offset(
    box_xyxy: Tuple[float, float, float, float],
    img_w: int,
    z_m: float,
    hfov_deg: float = 65.0,
) -> float:
    """Calculate lateral ground offset X (in meters) from horizontal center of bounding box."""
    x1, _, x2, _ = box_xyxy
    u_c = (x1 + x2) / 2.0
    u_0 = img_w / 2.0

    half_fov_rad = math.radians(hfov_deg / 2.0)
    f_u = (img_w / 2.0) / math.tan(half_fov_rad)

    # Offset from principal axis
    x_offset_m = ((u_c - u_0) / f_u) * z_m
    return round(x_offset_m, 2)


def google_maps_url(lat: float, lon: float) -> str:
    """Generate universal Google Maps search URL."""
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def google_maps_dir_url(lat: float, lon: float) -> str:
    """Generate Google Maps turn-by-turn navigation URL towards the hazard."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.6f},{lon:.6f}"


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two GPS coordinates in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c
