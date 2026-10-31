"""GV-YOLO26 layers, losses and trainers. Importing this package registers the layers with Ultralytics."""

from .register import register

register()

from .blocks import GSBottleneckC, GSConv, SPDConv, VoVGSCSPC  # noqa: E402
from .head import LGDH  # noqa: E402
from .loss import NWDDetectionLoss, nwd  # noqa: E402
from .model import GVDetectionModel  # noqa: E402
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
    "SuspensionPreview",
    "GVTrainer",
    "NWDTrainer",
    "DistillTrainer",
)
