"""Cheap convolution blocks used in the GV-YOLO26 backbone and neck."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv


class GSConv(nn.Module):
    """Dense half from a standard convolution, cheap half from a depthwise convolution, then a channel shuffle."""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, act=act)  # dense half
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)  # depthwise half

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.shape  # channel shuffle: interleave the dense and cheap halves
        return x2.view(b, 2, n // 2, h, w).transpose(1, 2).reshape(b, n, h, w)


class GSBottleneckC(nn.Module):
    """Two GSConv layers with a cheap depthwise shortcut."""

    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GSConv(c1, c_, 1, 1), GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = DWConv(c1, c2, k, s, act=False)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class VoVGSCSPC(nn.Module):
    """Cross-stage partial block with GSConv bottlenecks and one-shot aggregation."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneckC(c_, c_) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.cv2(x), self.gsb(self.cv1(x))), 1))
