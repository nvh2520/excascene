"""Fixed downstream decision layer: dig-point selection from predicted affordance maps.

All perception variants are evaluated through the SAME selector so that differences in regret are
attributable to perception quality (decision-centric evaluation).
"""
from __future__ import annotations

import numpy as np

from .sim.scene import compose_affordance, SEM_TERRAIN


def composed_affordance(pred: dict, reach: np.ndarray, z_target: float) -> np.ndarray:
    """Affordance composed from predicted component maps (height, semantics, material, hazard)."""
    return compose_affordance(pred["H"], pred["sem_label"], pred["mat_label"], pred["haz_prob"], reach, z_target)


def select_dig_point(score: np.ndarray, reach: np.ndarray, haz_prob: np.ndarray | None = None,
                     sigma: np.ndarray | None = None, lam: float = 0.0):
    """Return (row, col) of the argmax of a risk-penalised score restricted to reachable, predicted-safe cells."""
    s = score.astype(np.float64).copy()
    if sigma is not None and lam > 0:
        s = s - lam * sigma
    s[reach < 0.05] = -np.inf
    if haz_prob is not None:
        s[haz_prob > 0.5] = -np.inf
    if not np.isfinite(s).any():
        s = np.where(reach >= 0.05, score, -np.inf)
    idx = int(np.argmax(s))
    return np.unravel_index(idx, s.shape), s


def regret(aff_true: np.ndarray, pick) -> tuple:
    best = float(aff_true.max())
    got = float(aff_true[pick])
    r = best - got
    rn = r / best if best > 0.05 else 0.0
    return r, rn


def topk_regret(aff_true: np.ndarray, score: np.ndarray, k: int = 5) -> float:
    flat = score.ravel()
    ok = np.isfinite(flat)
    if ok.sum() < k:
        return float("nan")
    top = np.argpartition(-np.where(ok, flat, -np.inf), k)[:k]
    truth_top = np.sort(aff_true.ravel())[::-1][:k].mean()
    return float(truth_top - aff_true.ravel()[top].mean())


def unsafe_pick(gt_haz: np.ndarray, gt_sem: np.ndarray, pick) -> bool:
    return bool(gt_haz[pick] > 0.5 or gt_sem[pick] != SEM_TERRAIN)


def aurc(confidence: np.ndarray, risk: np.ndarray) -> tuple:
    """Area under the risk-coverage curve (lower is better) + risk at 50% and 80% coverage.
    confidence: higher = more confident; risk: per-frame regret."""
    order = np.argsort(-confidence)
    r = risk[order]
    cum = np.cumsum(r) / (np.arange(len(r)) + 1)
    cov = (np.arange(len(r)) + 1) / len(r)
    area = float(np.trapezoid(cum, cov)) if hasattr(np, "trapezoid") else float(np.trapz(cum, cov))
    r50 = float(cum[int(0.5 * len(r)) - 1]) if len(r) >= 2 else float("nan")
    r80 = float(cum[int(0.8 * len(r)) - 1]) if len(r) >= 2 else float("nan")
    return area, r50, r80
