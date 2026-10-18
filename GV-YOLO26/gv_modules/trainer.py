"""Trainers for GV-YOLO26: plain, NWD-loss and distillation."""

import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, RANK

from .model import _KD, GVDetectionModel


class GVTrainer(DetectionTrainer):
    """Builds a GVDetectionModel with the standard YOLO26 loss (ablation stages 1-5)."""

    use_nwd = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = GVDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        model.use_nwd = self.use_nwd
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model


class NWDTrainer(GVTrainer):
    """GVTrainer with the IoU + NWD box loss (ablation stage 6, the final GV-YOLO26)."""

    use_nwd = True


class DistillTrainer(NWDTrainer):
    """NWDTrainer that adds BCKD + CWD distillation from a frozen teacher. Set `teacher` before training."""

    teacher = None  # path to the trained teacher checkpoint (e.g. GV-YOLO26 at S scale)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if not self.teacher:
            raise ValueError("DistillTrainer.teacher must point to a trained teacher checkpoint")
        t, _ = load_checkpoint(self.teacher, device="cpu", fuse=False)
        self._teacher = t.float().eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        s_ch = [b[0].conv.in_channels for b in model.model[-1].stem]
        t_ch = [b[0].conv.in_channels for b in self._teacher.model[-1].stem]
        # 1x1 adapters map student neck features to teacher channels for CWD; trained with the student, never saved
        model.kd_adapters = nn.ModuleList(nn.Conv2d(cs, ct, 1) for cs, ct in zip(s_ch, t_ch))
        return model

    def _setup_train(self):
        super()._setup_train()
        _KD["teacher"] = self._teacher.to(self.device)
        LOGGER.info(f"Distilling from {self.teacher}")

    def save_model(self):
        ema = self.ema.ema
        adapters = ema._modules.pop("kd_adapters", None)  # keep the adapters out of the checkpoint
        try:
            super().save_model()
        finally:
            if adapters is not None:
                ema.kd_adapters = adapters

    def final_eval(self):
        _KD["teacher"] = None
        return super().final_eval()
