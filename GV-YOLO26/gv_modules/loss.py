"""NWD box loss and BCKD + CWD distillation losses for GV-YOLO26."""

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox, make_anchors


def nwd(pred, target, C=12.8, eps=1e-7):
    """Normalized Gaussian Wasserstein Distance between xyxy boxes given in pixels (equations 4.2 and 4.3)."""
    pc = (pred[:, :2] + pred[:, 2:]) / 2
    tc = (target[:, :2] + target[:, 2:]) / 2
    pwh = (pred[:, 2:] - pred[:, :2]).clamp(min=eps)
    twh = (target[:, 2:] - target[:, :2]).clamp(min=eps)
    d_center = ((pc - tc) ** 2).sum(dim=1)
    d_size = (((pwh - twh) / 2) ** 2).sum(dim=1)
    w2 = d_center + d_size + eps  # squared Wasserstein-2 distance
    return torch.exp(-torch.sqrt(w2) / C)


class NWDBboxLoss(BboxLoss):
    """YOLO26 box loss with the IoU term blended with NWD: L_box = alpha * L_IoU + (1 - alpha) * (1 - NWD)."""

    def __init__(self, reg_max=16, alpha=0.5):
        super().__init__(reg_max)
        self.alpha = alpha

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        loss_iou, loss_l1 = super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                            target_scores_sum, fg_mask, imgsz, stride)
        b, a = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[b, a].sum(-1)
        s = stride[a]  # boxes arrive in grid units; NWD's constant C is defined in pixels
        sim = nwd(pred_bboxes[b, a].float() * s, target_bboxes[b, a].float() * s)
        loss_nwd = ((1.0 - sim) * weight).sum() / target_scores_sum
        return self.alpha * loss_iou + (1 - self.alpha) * loss_nwd, loss_l1


class NWDDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss whose box term uses NWDBboxLoss."""

    alpha = 0.70

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.bbox_loss = NWDBboxLoss(self.reg_max, self.alpha).to(self.device)


def bckd_loss(s, t, stride, alpha=0.5):
    """Classification and localisation distillation on the head outputs (BCKD style).

    Guarantees coordinate positivity before dist2bbox decoding and bounds regression loss
    with hybrid clamped CIoU and Normalized Wasserstein Distance (NWD).
    s, t: one-to-many prediction dicts of the student and teacher ({"boxes", "scores", "feats"}).
    """
    t_prob = t["scores"].float().sigmoid()
    s_logit = s["scores"].float()
    w = (t_prob - s_logit.sigmoid()).abs().pow(2)  # focus on locations where the two disagree
    cls = (F.binary_cross_entropy_with_logits(s_logit, t_prob, reduction="none") * w).sum()
    cls = cls / t_prob.sum().clamp(min=1)

    anchors, stride_tensor = make_anchors(s["feats"], stride, 0.5)

    # 1. Coordinate Positivity Clamping: project raw distances via softplus + clamp
    # to guarantee positive distances from anchor centers: w = lt_x + rb_x > 0, h = lt_y + rb_y > 0
    s_dist = F.softplus(s["boxes"].float().permute(0, 2, 1), beta=1.0).clamp(min=1e-4)
    t_dist = F.softplus(t["boxes"].float().permute(0, 2, 1), beta=1.0).clamp(min=1e-4)

    sb = dist2bbox(s_dist, anchors, xywh=False)
    tb = dist2bbox(t_dist, anchors, xywh=False)

    # 2. Strict coordinate ordering: guarantee x2 > x1 and y2 > y1
    sb_x1 = torch.min(sb[..., 0], sb[..., 2])
    sb_y1 = torch.min(sb[..., 1], sb[..., 3])
    sb_x2 = torch.max(sb[..., 0], sb[..., 2]).clamp(min=sb_x1 + 1e-2)
    sb_y2 = torch.max(sb[..., 1], sb[..., 3]).clamp(min=sb_y1 + 1e-2)
    sb_clamped = torch.stack([sb_x1, sb_y1, sb_x2, sb_y2], dim=-1)

    tb_x1 = torch.min(tb[..., 0], tb[..., 2])
    tb_y1 = torch.min(tb[..., 1], tb[..., 3])
    tb_x2 = torch.max(tb[..., 0], tb[..., 2]).clamp(min=tb_x1 + 1e-2)
    tb_y2 = torch.max(tb[..., 1], tb[..., 3]).clamp(min=tb_y1 + 1e-2)
    tb_clamped = torch.stack([tb_x1, tb_y1, tb_x2, tb_y2], dim=-1)

    # 3. Teacher foreground masking: distill coordinates only on confident teacher predictions
    if t_prob.shape[1] == anchors.shape[0]:  # (bs, anchors, nc)
        wt = t_prob.max(dim=-1)[0]
    else:  # (bs, nc, anchors)
        wt = t_prob.max(dim=1)[0]

    mask = wt > 0.25
    if mask.any():
        # Bounded CIoU in [-1.0, 1.0], giving (1.0 - ciou) in [0.0, 2.0]
        ciou = bbox_iou(sb_clamped[mask], tb_clamped[mask], xywh=False, CIoU=True).squeeze(-1).clamp(min=-1.0, max=1.0)
        loss_iou = 1.0 - ciou

        # Bounded Normalized Wasserstein Distance in pixel coordinates: L_NWD in [0.0, 1.0)
        st = stride_tensor.squeeze(-1) if stride_tensor.dim() > 1 else stride_tensor
        if mask.dim() == 2:
            st = st.unsqueeze(0).expand(mask.shape[0], -1)
        s_stride_mask = st[mask].unsqueeze(-1)
        sim_nwd = nwd(sb_clamped[mask] * s_stride_mask, tb_clamped[mask] * s_stride_mask)
        loss_nwd = 1.0 - sim_nwd.clamp(min=1e-7, max=1.0)

        # Hybrid bounded box distillation loss
        loss_box_distill = (1.0 - alpha) * loss_iou + alpha * loss_nwd
        box = (loss_box_distill * wt[mask]).sum() / wt[mask].sum().clamp(min=1)
    else:
        box = torch.zeros((), device=sb.device)
    return cls + box


def cwd_loss(s_feats, t_feats, adapters, tau=1.0):
    """Channel-wise distillation: per-channel spatial softmax, KL(teacher || student)."""
    total = 0.0
    for sf, tf, ad in zip(s_feats, t_feats, adapters):
        sf = ad(sf).float()
        tf = tf.float()
        n, c, h, w = sf.shape
        log_ps = F.log_softmax(sf.reshape(n * c, h * w) / tau, dim=1)
        pt = F.softmax(tf.reshape(n * c, h * w) / tau, dim=1)
        total = total + (pt * (torch.log(pt + 1e-12) - log_ps)).sum() * tau**2 / (n * c)
    return total / len(s_feats)
