"""Per-frame metrics (NumPy).  Everything returns plain floats so results can be stored as CSV rows."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .sim.scene import N_BEV_SEM, SEM_TERRAIN
from .decision import (composed_affordance, select_dig_point, regret, topk_regret, unsafe_pick)

N_ECE_BINS = 15
Z90 = 1.6449


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney); score higher => predicts label==1."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece_bins(conf: np.ndarray, correct: np.ndarray):
    """Per-bin sums (conf_sum, acc_sum, count) so ECE can be aggregated over frames."""
    b = np.clip((conf * N_ECE_BINS).astype(int), 0, N_ECE_BINS - 1)
    cnt = np.bincount(b, minlength=N_ECE_BINS).astype(np.float64)
    csum = np.bincount(b, weights=conf, minlength=N_ECE_BINS)
    asum = np.bincount(b, weights=correct.astype(np.float64), minlength=N_ECE_BINS)
    return csum, asum, cnt


def ece_from_bins(csum, asum, cnt) -> float:
    n = cnt.sum()
    if n == 0:
        return float("nan")
    m = cnt > 0
    return float((np.abs(csum[m] / cnt[m] - asum[m] / cnt[m]) * cnt[m]).sum() / n)


def frame_metrics(pred: dict, gt: dict, conf: dict | None, lam: float = 1.0) -> dict:
    """pred: H, H_sigma (or None), sem_prob (C,N,N), mat_label, haz_prob, aff (or None), aff_sigma (or None)
    gt: H, sem, mat, haz, aff, reach, vis, z_target
    conf: conformal quantiles dict(q_H, q_aff) or None."""
    m = {}
    vis = gt["vis"] > 0.5
    reach = gt["reach"] > 0.05
    err = np.abs(pred["H"] - gt["H"])
    m["H_mae_vis"] = float(err[vis].mean()) if vis.any() else float("nan")
    m["H_mae_all"] = float(err.mean())
    m["H_mae_reach"] = float(err[reach].mean()) if reach.any() else float("nan")
    sem_label = pred["sem_prob"].argmax(0)
    m["sem_acc_vis"] = float((sem_label == gt["sem"])[vis].mean()) if vis.any() else float("nan")
    # confusion (flattened) on visible cells for suite-level mIoU
    cm = np.bincount((gt["sem"][vis] * N_BEV_SEM + sem_label[vis]).ravel(), minlength=N_BEV_SEM ** 2)
    for i in range(N_BEV_SEM ** 2):
        m[f"cm_{i}"] = int(cm[i])
    ter = (gt["sem"] == SEM_TERRAIN) & vis
    m["mat_acc"] = float((pred["mat_label"] == gt["mat"])[ter].mean()) if ter.any() else float("nan")
    hp = pred["haz_prob"] > 0.5
    hg = gt["haz"] > 0.5
    inter, union = (hp & hg).sum(), (hp | hg).sum()
    m["haz_iou"] = float(inter / union) if union > 0 else float("nan")
    m["haz_miss_rate"] = float((~hp & hg).sum() / max(hg.sum(), 1))
    m["haz_brier"] = float(((pred["haz_prob"] - gt["haz"]) ** 2).mean())
    # --- decisions: composed selector (always available) and direct-head selector (if present)
    comp = composed_affordance(dict(H=pred["H"], sem_label=sem_label, mat_label=pred["mat_label"], haz_prob=pred["haz_prob"]),
                               gt["reach"], gt["z_target"])
    m["affc_mae"] = float(np.abs(comp - gt["aff"])[reach].mean()) if reach.any() else float("nan")
    pick, score = select_dig_point(comp, gt["reach"], haz_prob=None)
    m["regret_comp"], m["regret_comp_norm"] = regret(gt["aff"], pick)
    m["unsafe_comp"] = float(unsafe_pick(gt["haz"], gt["sem"], pick))
    m["top5_regret_comp"] = topk_regret(gt["aff"], score)
    m["conf_comp"] = float(np.nanmax(np.where(np.isfinite(score), score, np.nan))) if np.isfinite(score).any() else 0.0
    if pred.get("aff") is not None:
        a = pred["aff"]
        m["aff_mae"] = float(np.abs(a - gt["aff"])[reach].mean()) if reach.any() else float("nan")
        if reach.sum() > 10 and np.std(gt["aff"][reach]) > 1e-6:
            m["aff_spearman"] = float(spearmanr(a[reach], gt["aff"][reach]).correlation)
        pick, score = select_dig_point(a, gt["reach"], haz_prob=pred["haz_prob"])
        m["regret_direct"], m["regret_direct_norm"] = regret(gt["aff"], pick)
        m["unsafe_direct"] = float(unsafe_pick(gt["haz"], gt["sem"], pick))
        m["top5_regret_direct"] = topk_regret(gt["aff"], score)
        m["conf_direct"] = float(np.nanmax(np.where(np.isfinite(score), score, np.nan))) if np.isfinite(score).any() else 0.0
        if pred.get("aff_sigma") is not None:
            pick, score = select_dig_point(a, gt["reach"], haz_prob=pred["haz_prob"], sigma=pred["aff_sigma"], lam=lam)
            m["regret_risk"], m["regret_risk_norm"] = regret(gt["aff"], pick)
            m["unsafe_risk"] = float(unsafe_pick(gt["haz"], gt["sem"], pick))
            m["conf_risk"] = float(np.nanmax(np.where(np.isfinite(score), score, np.nan))) if np.isfinite(score).any() else 0.0
            # is the chosen cell inside the conformal lower-bound "safe-diggable" set?
            if conf is not None:
                lb = a - conf["q_aff"] * pred["aff_sigma"]
                m["cp_aff_cov"] = float((np.abs(a - gt["aff"]) <= conf["q_aff"] * pred["aff_sigma"])[reach].mean())
                m["cp_aff_width"] = float((2 * conf["q_aff"] * pred["aff_sigma"])[reach].mean())
                m["lb_pick_true_aff"] = float(gt["aff"][np.unravel_index(int(np.argmax(np.where(reach, lb, -np.inf))), lb.shape)])
    # --- calibration of the semantic head (visible cells)
    conf_s = pred["sem_prob"].max(0)
    csum, asum, cnt = ece_bins(conf_s[vis], (sem_label == gt["sem"])[vis]) if vis.any() else (np.zeros(N_ECE_BINS),) * 3
    for i in range(N_ECE_BINS):
        m[f"ece_c{i}"], m[f"ece_a{i}"], m[f"ece_n{i}"] = float(csum[i]), float(asum[i]), float(cnt[i])
    # --- height uncertainty
    big = (err > 0.25).astype(int)
    if pred.get("H_sigma") is not None:
        s = pred["H_sigma"]
        z = err / np.maximum(s, 1e-3)
        m["H_nll_vis"] = float((0.5 * z[vis] ** 2 + np.log(s[vis]) + 0.5 * np.log(2 * np.pi)).mean()) if vis.any() else float("nan")
        m["H_cov90_vis"] = float((z[vis] <= Z90).mean()) if vis.any() else float("nan")
        m["H_cov90_all"] = float((z <= Z90).mean())
        m["H_sigma_mean_vis"] = float(s[vis].mean()) if vis.any() else float("nan")
        m["H_sigma_mean_unvis"] = float(s[~vis].mean()) if (~vis).any() else float("nan")
        m["err_auroc"] = auroc(s.ravel(), big.ravel())
        if conf is not None:
            m["cp_H_cov"] = float((err <= conf["q_H"] * s)[vis].mean()) if vis.any() else float("nan")
            m["cp_H_width"] = float((2 * conf["q_H"] * s)[vis].mean()) if vis.any() else float("nan")
    else:
        # uncertainty-free models: constant-width conformal intervals from the calibration residuals
        if conf is not None:
            m["cp_H_cov"] = float((err <= conf["q_H"])[vis].mean()) if vis.any() else float("nan")
            m["cp_H_width"] = float(2 * conf["q_H"])
        # entropy of the semantic head as a generic uncertainty proxy for error detection
        ent = -(pred["sem_prob"] * np.log(pred["sem_prob"] + 1e-9)).sum(0)
        m["err_auroc"] = auroc(ent.ravel(), big.ravel())
    if pred.get("gate") is not None:
        for i, g in enumerate(pred["gate"]):
            m[f"gate_{i}"] = float(g)
    return m
