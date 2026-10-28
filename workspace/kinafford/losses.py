"""Multi-task losses.  Cells are weighted 1.0 if any sensor observed them and 0.3 otherwise, so the
network still learns to extrapolate (with high uncertainty when heteroscedastic heads are used)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .sim.scene import SEM_TERRAIN
from .sim.kinematics import SENSOR_ORDER

DEFAULT_WEIGHTS = dict(H=2.0, sem=1.0, mat=0.5, haz=1.0, aff=2.0, seg=0.3, gate=0.2)


def gaussian_nll(mu, logvar, y, w):
    nll = 0.5 * (torch.exp(-logvar) * (y - mu) ** 2 + logvar)
    return (w * nll).sum() / w.sum().clamp(min=1.0)


def compute_losses(out: dict, batch: dict, model_cfg: dict, weights: dict | None = None) -> dict:
    W = dict(DEFAULT_WEIGHTS)
    if weights:
        W.update(weights)
    vis = batch["gt_vis"]
    w = 0.3 + 0.7 * vis
    losses = {}
    # height
    err = out["H"] - batch["gt_H"]
    if "H_logvar" in out:
        losses["H"] = gaussian_nll(out["H"], out["H_logvar"], batch["gt_H"], w) + 0.1 * (w * err.abs()).sum() / w.sum()
    else:
        losses["H"] = (w * err.abs()).sum() / w.sum()
    # semantics / material / hazard
    ce = F.cross_entropy(out["sem"], batch["gt_sem"], reduction="none")
    losses["sem"] = (w * ce).sum() / w.sum()
    mmask = (batch["gt_sem"] == SEM_TERRAIN).float() * w
    cem = F.cross_entropy(out["mat"], batch["gt_mat"], reduction="none")
    losses["mat"] = (mmask * cem).sum() / mmask.sum().clamp(min=1.0)
    bce = F.binary_cross_entropy_with_logits(out["haz"], batch["gt_haz"], reduction="none")
    losses["haz"] = (w * bce).sum() / w.sum()
    # affordance (direct head)
    if "aff" in out:
        wa = w * (0.2 + 0.8 * (batch["gt_reach"] > 0.05).float())
        a_err = torch.sigmoid(out["aff"]) - batch["gt_aff"]
        if "aff_logvar" in out:
            losses["aff"] = gaussian_nll(torch.sigmoid(out["aff"]), out["aff_logvar"], batch["gt_aff"], wa) + 0.5 * (wa * a_err.abs()).sum() / wa.sum()
        else:
            losses["aff"] = (wa * a_err.abs()).sum() / wa.sum()
    # image-space semantic auxiliary loss (stride-4 targets)
    if out.get("img_seg"):
        seg_losses = []
        for k, logits in out["img_seg"]:
            tgt = batch["img_sem"][:, k, 2::4, 2::4][:, :logits.shape[-2], :logits.shape[-1]]
            l = F.cross_entropy(logits.float(), tgt, reduction="none").mean((1, 2))
            seg_losses.append((l * batch["cam_present"][:, k]).sum() / batch["cam_present"][:, k].sum().clamp(min=1.0))
        losses["seg"] = torch.stack(seg_losses).mean()
    # reliability gate supervision
    if "gate" in out:
        idx = [SENSOR_ORDER.index(s) for s in model_cfg["sensors"]]
        losses["gate"] = F.mse_loss(out["gate"], batch["rel_gt"][:, idx])
    total = sum(W[k] * v for k, v in losses.items())
    losses["total"] = total
    return losses
