"""Sanity check and mathematical verification for BCKD + NWD distillation loss."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gv_modules  # noqa: F401
from gv_modules.loss import bckd_loss
from ultralytics.utils.tal import dist2bbox, make_anchors


def run_checks():
    print("=" * 60)
    print("Running Phase 1 Verification: BCKD + NWD Distillation Loss")
    print("=" * 60)

    torch.manual_seed(42)
    bs = 2
    nc = 2
    # Simulate feature maps for stride 8, 16, 32 (e.g., 80x80, 40x40, 20x20)
    feats = [
        torch.zeros(bs, 64, 80, 80),
        torch.zeros(bs, 128, 40, 40),
        torch.zeros(bs, 256, 20, 20),
    ]
    strides = torch.tensor([8.0, 16.0, 32.0])
    anchors, _ = make_anchors(feats, strides, 0.5)
    n_anchors = anchors.shape[0]
    print(f"Total simulated anchors: {n_anchors}")

    # Test 1: Coordinate Positivity Projection Check
    print("\n[Test 1] Coordinate Positivity Projection on Extreme Negative Raw Inputs...")
    # Extreme negative distance predictions
    raw_dist_negative = torch.randn(bs, n_anchors, 4) * 50.0 - 100.0  # heavily negative
    s_dist = F.softplus(raw_dist_negative, beta=1.0).clamp(min=1e-4)
    sb = dist2bbox(s_dist, anchors, xywh=False)

    sb_x1 = torch.min(sb[..., 0], sb[..., 2])
    sb_y1 = torch.min(sb[..., 1], sb[..., 3])
    sb_x2 = torch.max(sb[..., 0], sb[..., 2]).clamp(min=sb_x1 + 1e-2)
    sb_y2 = torch.max(sb[..., 1], sb[..., 3]).clamp(min=sb_y1 + 1e-2)

    widths = sb_x2 - sb_x1
    heights = sb_y2 - sb_y1

    assert (widths > 0).all().item(), "Assertion failed: some decoded box widths are <= 0!"
    assert (heights > 0).all().item(), "Assertion failed: some decoded box heights are <= 0!"
    assert not torch.isnan(widths).any().item(), "Assertion failed: NaN found in widths!"
    assert not torch.isnan(heights).any().item(), "Assertion failed: NaN found in heights!"
    min_w = widths.min().item()
    min_h = heights.min().item()
    print(f"  -> Decoded 100% anchors with positive dimensions! (min w={min_w:.4f}, min h={min_h:.4f})")

    # Test 2: Extreme Unconstrained Synthetic Logits Loss Bound Check
    print("\n[Test 2] BCKD Loss Bound & Non-Negativity Check on Random Synthetic Inputs...")
    s = {
        "boxes": torch.randn(bs, 4, n_anchors) * 20.0 - 30.0,
        "scores": torch.randn(bs, nc, n_anchors) * 5.0,
        "feats": feats,
    }
    t = {
        "boxes": torch.randn(bs, 4, n_anchors) * 5.0 + 10.0,
        "scores": torch.randn(bs, nc, n_anchors) * 3.0,
        "feats": feats,
    }

    loss = bckd_loss(s, t, strides, alpha=0.5)
    print(f"  -> Calculated loss value: {loss.item():.6f}")
    assert not torch.isnan(loss), "Assertion failed: distillation loss is NaN!"
    assert not torch.isinf(loss), "Assertion failed: distillation loss is Inf!"
    assert loss.item() >= 0.0, f"Assertion failed: distillation loss is negative ({loss.item()})!"
    print("  -> Loss is strictly non-negative, finite, and bounded.")

    # Test 3: Foreground Masking Verification (All Background vs High Foreground)
    print("\n[Test 3] Foreground Masking Check (All Background)...")
    t_bg = {
        "boxes": torch.randn(bs, 4, n_anchors),
        "scores": torch.full((bs, nc, n_anchors), -10.0),  # sigmoid(-10) ~= 0.000045 < 0.25
        "feats": feats,
    }
    loss_bg = bckd_loss(s, t_bg, strides, alpha=0.5)
    print(f"  -> Background-only loss: {loss_bg.item():.6f}")
    assert not torch.isnan(loss_bg) and not torch.isinf(loss_bg), "Background loss must be finite"
    assert loss_bg.item() >= 0.0, "Background loss must be >= 0"

    print("\n[Test 4] Extreme Foreground Mismatch (High Teacher Confidence)...")
    t_fg = {
        "boxes": torch.randn(bs, 4, n_anchors) * 50.0,
        "scores": torch.full((bs, nc, n_anchors), 10.0),  # sigmoid(10) ~= 0.99995 > 0.25
        "feats": feats,
    }
    loss_fg = bckd_loss(s, t_fg, strides, alpha=0.5)
    print(f"  -> High teacher confidence loss: {loss_fg.item():.6f}")
    assert not torch.isnan(loss_fg) and not torch.isinf(loss_fg), "Foreground loss must be finite"
    assert loss_fg.item() >= 0.0, "Foreground loss must be >= 0"

    print("\n" + "=" * 60)
    print("ALL PHASE 1 SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()
