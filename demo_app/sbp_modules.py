"""SBP-YOLO architectural modules and runtime registration.

Provides standalone definitions for VoVGSCSPC, GSConv, GSBottleneckC, and Detect_Efficient
(LEDH head) so that SBP-YOLO checkpoints can be unpickled and run with standard Ultralytics.
"""
import math
import sys
import types
from pathlib import Path
import torch
import torch.nn as nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv, autopad
from ultralytics.nn.modules.head import Detect
import ultralytics.nn.tasks as tasks
from ultralytics.utils.tal import dist2bbox, make_anchors


class GSConv(nn.Module):
    """Dense half from a standard convolution, cheap half from a depthwise convolution, channel-shuffled."""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.shape
        return x2.view(b, 2, n // 2, h, w).transpose(1, 2).reshape(b, n, h, w)


class GSBottleneck(nn.Module):
    """GS Bottleneck module matching ultralytics-distill naming (conv_lighting)."""

    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False)
        )
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class GSBottleneckC(GSBottleneck):
    """Cheap GS Bottleneck with depthwise convolution shortcut."""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)


class VoVGSCSP(nn.Module):
    """Base VoVGSCSP module with GSBottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class VoVGSCSPC(VoVGSCSP):
    """SlimNeck VoVGSCSPC cross-stage partial block."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * 0.5)
        self.gsb = GSBottleneckC(c_, c_, 1, 1)


class Detect_Efficient(Detect):
    """SBP-YOLO lightweight efficient detection head (LEDH)."""

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.stem = nn.ModuleList(
            nn.Sequential(Conv(x, x, 3, g=max(1, x // 16)), Conv(x, x, 3, g=max(1, x // 16)))
            for x in ch
        )
        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and getattr(self, "format", None) in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a.bias.data[:] = 1.0
            b.bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


def register_sbp_modules():
    """Register synthetic extra_modules into sys.modules and tasks namespace."""
    extra_pkg = types.ModuleType("ultralytics.nn.extra_modules")
    extra_block = types.ModuleType("ultralytics.nn.extra_modules.block")
    extra_head = types.ModuleType("ultralytics.nn.extra_modules.head")

    extra_block.Conv = Conv
    extra_block.DWConv = DWConv
    extra_block.GSConv = GSConv
    extra_block.GSBottleneck = GSBottleneck
    extra_block.GSBottleneckC = GSBottleneckC
    extra_block.VoVGSCSP = VoVGSCSP
    extra_block.VoVGSCSPC = VoVGSCSPC

    extra_head.Detect_Efficient = Detect_Efficient

    extra_pkg.block = extra_block
    extra_pkg.head = extra_head

    sys.modules["ultralytics.nn.extra_modules"] = extra_pkg
    sys.modules["ultralytics.nn.extra_modules.block"] = extra_block
    sys.modules["ultralytics.nn.extra_modules.head"] = extra_head

    setattr(tasks, "Detect_Efficient", Detect_Efficient)
    setattr(tasks, "VoVGSCSPC", VoVGSCSPC)
    setattr(tasks, "GSConv", GSConv)
    setattr(tasks, "GSBottleneck", GSBottleneck)
    setattr(tasks, "GSBottleneckC", GSBottleneckC)

    if hasattr(torch.serialization, "add_safe_globals"):
        try:
            torch.serialization.add_safe_globals([
                Detect_Efficient, VoVGSCSPC, GSBottleneckC, GSBottleneck, GSConv, Conv, DWConv
            ])
        except Exception:
            pass
