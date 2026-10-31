"""GV-YOLO26 detection model: Ultralytics DetectionModel with the NWD criterion and an optional distillation term."""

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

from .loss import NWDDetectionLoss, bckd_loss, cwd_loss

# The teacher lives here rather than on the model, so it is never deep-copied into the EMA or pickled into checkpoints.
_KD = {"teacher": None, "weight_bckd": 0.5, "weight_cwd": 0.5, "alpha": 0.5, "warmup_factor": 1.0}


class GVDetectionModel(DetectionModel):
    use_nwd = False

    def init_criterion(self):
        loss_fn = NWDDetectionLoss if self.use_nwd else v8DetectionLoss
        end2end = getattr(self.model[-1], "one2one_cv2", None) is not None
        return E2ELoss(self, loss_fn) if end2end else loss_fn(self)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        loss, items = self.criterion(preds, batch)

        teacher = _KD["teacher"]
        if teacher is None:
            return loss, items
        kd = torch.zeros((), device=loss.device)
        if self.training and hasattr(self, "kd_adapters"):
            s = preds[1] if isinstance(preds, tuple) else preds
            s = s["one2many"] if "one2many" in s else s
            with torch.no_grad():
                t = teacher(batch["img"])[1]  # teacher runs in eval mode: (y, preds)
                t = t["one2many"] if "one2many" in t else t
            warmup = _KD.get("warmup_factor", 1.0)
            alpha_bckd = _KD.get("alpha", 0.5)
            kd = warmup * (
                _KD["weight_bckd"] * bckd_loss(s, t, self.stride, alpha=alpha_bckd)
                + _KD["weight_cwd"] * cwd_loss(s["feats"], t["feats"], self.kd_adapters)
            )
            loss = torch.cat([loss, (kd * batch["img"].shape[0]).view(1).to(loss.dtype)])
        items = {**items, "kd_loss": kd.detach()}  # same keys in training and validation
        return loss, items
