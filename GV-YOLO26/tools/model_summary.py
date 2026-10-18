"""Build every GV-YOLO26 design stage, check its outputs and print parameters and GFLOPs (Table 5.3 columns)."""

import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gv_modules  # noqa: E402,F401  registers custom layers
from gv_modules import GVDetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("1 YOLO26n baseline", "yolo26n.yaml"),
    ("2 + P2 head", ROOT / "models/ablation/gv-yolo26n-p2.yaml"),
    ("3 + LGDH", ROOT / "models/ablation/gv-yolo26n-p2-lgdh.yaml"),
    ("4 + GhostConv", ROOT / "models/ablation/gv-yolo26n-p2-lgdh-ghost.yaml"),
    ("5 + VoVGSCSPC (GV-YOLO26)", ROOT / "models/gv-yolo26n.yaml"),
    ("teacher GV-YOLO26-S", ROOT / "models/gv-yolo26s.yaml"),
]


def summary(cfg, imgsz=640):
    m = GVDetectionModel(str(cfg), nc=2, verbose=False)
    params = sum(p.numel() for p in m.parameters())
    flops = get_flops(m, imgsz)
    m.train()
    out = m(torch.zeros(1, 3, imgsz, imgsz))
    n_anchor = out["one2many"]["boxes"].shape[-1]
    m.eval()
    y = m(torch.zeros(1, 3, imgsz, imgsz))[0]
    f = deepcopy(m).fuse(verbose=False)  # deployed form: BN folded, one-to-many branch removed
    fparams = sum(p.numel() for p in f.parameters())
    return params, flops, fparams, get_flops(f, imgsz), n_anchor, tuple(y.shape)


if __name__ == "__main__":
    print(f"{'stage':32}{'params':>11}{'GFLOPs':>8}{'fused params':>14}{'fused GFLOPs':>14}{'anchors':>9}  output")
    for name, cfg in STAGES:
        p, g, fp, fg, a, y = summary(cfg)
        print(f"{name:32}{p / 1e6:>10.2f}M{g:>8.1f}{fp / 1e6:>13.2f}M{fg:>14.1f}{a:>9}  {y}")
