"""GV-YOLO26 layers, losses and trainers. Importing this package registers the layers with Ultralytics."""

from .register import register

register()

from .blocks import GSBottleneckC, GSConv, SPDConv, VoVGSCSPC  # noqa: E402
from .head import LGDH  # noqa: E402
from .loss import NWDDetectionLoss, nwd  # noqa: E402
from .model import GVDetectionModel  # noqa: E402
from .preprocess import enhance_road_surface  # noqa: E402
from .hazard_mapper import (  # noqa: E402
    estimate_lateral_offset,
    google_maps_dir_url,
    google_maps_url,
    haversine_distance_m,
    project_hazard_gps,
)
from .suspension_preview import SuspensionPreview  # noqa: E402
from .trainer import DistillTrainer, GVTrainer, NWDTrainer  # noqa: E402

__all__ = (
    "GSConv",
    "GSBottleneckC",
    "SPDConv",
    "VoVGSCSPC",
    "LGDH",
    "nwd",
    "NWDDetectionLoss",
    "GVDetectionModel",
    "enhance_road_surface",
    "project_hazard_gps",
    "estimate_lateral_offset",
    "google_maps_url",
    "google_maps_dir_url",
    "haversine_distance_m",
    "SuspensionPreview",
    "GVTrainer",
    "NWDTrainer",
    "DistillTrainer",
)
