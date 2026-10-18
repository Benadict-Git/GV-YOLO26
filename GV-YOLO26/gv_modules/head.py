"""Lightweight Grouped Detection Head (LGDH) on top of the YOLO26 end-to-end Detect head."""

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


class LGDH(Detect):
    """Detect head whose box and class outputs share one grouped 3x3 stem per level.

    The one-to-many branch (training) and the one-to-one branch (NMS-free inference) each get their own copy of the
    stem and output convolutions, exactly as YOLO26 duplicates its default head.
    """

    channels_per_group = 8  # groups = c / 8 in each 3x3 stem convolution

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        g = lambda c: max(c // self.channels_per_group, 1)  # noqa: E731
        self.stem = nn.ModuleList(nn.Sequential(Conv(x, x, 3, g=g(x)), Conv(x, x, 3, g=g(x))) for x in ch)
        # single-layer Sequentials keep Detect.bias_init() (which touches [-1]) working unchanged
        self.cv2 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, self.nc, 1)) for x in ch)
        if end2end:
            self.one2one_stem = copy.deepcopy(self.stem)
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        return {"box_head": self.cv2, "cls_head": self.cv3, "stem": self.stem}

    @property
    def one2one(self):
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3, "stem": self.one2one_stem}

    def forward_head(self, x, box_head=None, cls_head=None, stem=None):
        if box_head is None or cls_head is None:  # branch removed by fuse()
            return {}
        bs = x[0].shape[0]
        f = [stem[i](x[i]) for i in range(self.nl)]  # shared stem, computed once per level
        boxes = torch.cat([box_head[i](f[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](f[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}
