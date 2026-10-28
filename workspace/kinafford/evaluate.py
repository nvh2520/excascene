"""Evaluate one trained run (or an ensemble of runs) on the calibration split and all suites.

    python -m kinafford.evaluate --run runs/R2_full/seed0 --tag R2_full --seed 0 --data data --out results
    python -m kinafford.evaluate --run runs/R2_full/seed0 runs/R2_full/seed1 runs/R2_full/seed2 --tag R2_full_ens --seed 0 ...

Writes  <out>/<tag>_s<seed>.csv  (one row per frame)  and  <out>/<tag>_s<seed>.json  (conformal quantiles, latency).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import ExcaSceneDataset, to_device
from .models.variants import build_model, count_params
from .metrics import frame_metrics
from .sim.degrade import SUITE_ORDER
from .sim.kinematics import EXCAVATOR

ALPHA = 0.10  # 90 % conformal coverage target


def load_runs(run_dirs, ckpt, device):
    models, cfgs = [], []
    for rd in run_dirs:
        state = torch.load(os.path.join(rd, f"{ckpt}.pt"), map_location="cpu")
        cfg = state["cfg"]
        m = build_model(cfg["model"])
        m.load_state_dict(state["model"])
        m.to(device).eval()
        models.append(m)
        cfgs.append(cfg)
    return models, cfgs


@torch.no_grad()
def predict(models, batch, amp=True):
    """Ensemble-aware prediction.  Returns dict of numpy arrays (batch-first)."""
    Hs, Hv, sems, mats, hazs, affs, affv, gates = [], [], [], [], [], [], [], []
    for m in models:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and batch["img"].is_cuda):
            out = m(batch)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        Hs.append(out["H"])
        Hv.append(torch.exp(out["H_logvar"]) if "H_logvar" in out else None)
        sems.append(torch.softmax(out["sem"], 1))
        mats.append(torch.softmax(out["mat"], 1))
        hazs.append(torch.sigmoid(out["haz"]))
        affs.append(torch.sigmoid(out["aff"]) if "aff" in out else None)
        affv.append(torch.exp(out["aff_logvar"]) if "aff_logvar" in out else None)
        gates.append(out.get("gate"))
    H = torch.stack(Hs).mean(0)
    H_sigma = None
    if Hv[0] is not None:
        var = torch.stack(Hv).mean(0) + (torch.stack(Hs).var(0, unbiased=False) if len(Hs) > 1 else 0.0)
        H_sigma = var.sqrt()
    sem = torch.stack(sems).mean(0)
    mat = torch.stack(mats).mean(0).argmax(1)
    haz = torch.stack(hazs).mean(0)
    aff, aff_sigma = None, None
    if affs[0] is not None:
        aff = torch.stack(affs).mean(0)
        if affv[0] is not None:
            var = torch.stack(affv).mean(0) + (torch.stack(affs).var(0, unbiased=False) if len(affs) > 1 else 0.0)
            aff_sigma = var.sqrt()
        elif len(affs) > 1:
            aff_sigma = torch.stack(affs).std(0, unbiased=False)
    gate = torch.stack(gates).mean(0) if gates[0] is not None else None
    f = lambda t: None if t is None else t.cpu().numpy()
    return dict(H=f(H), H_sigma=f(H_sigma), sem_prob=f(sem), mat_label=f(mat), haz_prob=f(haz), aff=f(aff),
                aff_sigma=f(aff_sigma), gate=f(gate))


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    k = min(math.ceil((n + 1) * (1 - alpha)), n)
    return float(np.sort(scores)[k - 1])


def calibrate(models, loader, device, amp, per_frame_cells=400, seed=0):
    rng = np.random.default_rng(seed)
    hs, af = [], []
    for batch in loader:
        b = to_device(batch, device)
        p = predict(models, b, amp)
        gt_H, vis = batch["gt_H"].numpy(), batch["gt_vis"].numpy() > 0.5
        gt_aff, reach = batch["gt_aff"].numpy(), batch["gt_reach"].numpy() > 0.05
        for i in range(gt_H.shape[0]):
            err = np.abs(p["H"][i] - gt_H[i])[vis[i]]
            if p["H_sigma"] is not None:
                err = err / np.maximum(p["H_sigma"][i][vis[i]], 1e-3)
            if len(err):
                hs.append(rng.choice(err, size=min(per_frame_cells, len(err)), replace=False))
            if p["aff"] is not None and p["aff_sigma"] is not None:
                ea = np.abs(p["aff"][i] - gt_aff[i])[reach[i]] / np.maximum(p["aff_sigma"][i][reach[i]], 1e-4)
                if len(ea):
                    af.append(rng.choice(ea, size=min(per_frame_cells, len(ea)), replace=False))
    q_H = conformal_quantile(np.concatenate(hs)) if hs else float("nan")
    q_aff = conformal_quantile(np.concatenate(af)) if af else float("nan")
    return dict(q_H=q_H, q_aff=q_aff, n_H=int(sum(len(x) for x in hs)), n_aff=int(sum(len(x) for x in af)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--lam", type=float, default=1.0, help="risk-aversion weight for the risk-aware selector")
    ap.add_argument("--save_preds", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"{args.tag}_s{args.seed}.csv")
    if os.path.exists(csv_path):
        print(f"[eval] {csv_path} exists, skipping")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(12_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
    amp = device.type == "cuda"
    models, cfgs = load_runs(args.run, args.ckpt, device)
    variant = cfgs[0]["variant"]
    print(f"[eval] tag={args.tag} runs={len(models)} variant={variant} params={count_params(models[0]) / 1e6:.2f}M")

    cal_dl = DataLoader(ExcaSceneDataset(args.data, "cal", limit=args.limit), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    conf = calibrate(models, cal_dl, device, amp, seed=args.seed)
    print(f"[eval] conformal: {conf}")

    suites = ["test_iid"] + [f"suite_{s}" for s in SUITE_ORDER]
    if args.suites:
        suites = args.suites
    q_nom = EXCAVATOR["q_nom"]
    rows, lat = [], []
    preds_dir = os.path.join(args.out, "preds", f"{args.tag}_s{args.seed}")
    for suite in suites:
        try:
            ds = ExcaSceneDataset(args.data, suite, limit=args.limit)
        except FileNotFoundError:
            print(f"[eval] suite {suite} missing, skipping")
            continue
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
        n_saved = 0
        for batch in dl:
            b = to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            p = predict(models, b, amp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.time() - t0) / batch["gt_H"].shape[0])
            B = batch["gt_H"].shape[0]
            for i in range(B):
                gt = dict(H=batch["gt_H"][i].numpy(), sem=batch["gt_sem"][i].numpy(), mat=batch["gt_mat"][i].numpy(),
                          haz=batch["gt_haz"][i].numpy(), aff=batch["gt_aff"][i].numpy(), reach=batch["gt_reach"][i].numpy(),
                          vis=batch["gt_vis"][i].numpy(), z_target=float(batch["z_target"][i]))
                pi = {k: (None if v is None else v[i]) for k, v in p.items()}
                m = frame_metrics(pi, gt, conf, lam=args.lam)
                qt = batch["q_true"][i].numpy()
                m.update(tag=args.tag, variant=variant, seed=args.seed, suite=suite.replace("suite_", ""),
                         scene_id=int(batch["scene_id"][i]), frame_id=int(batch["frame_id"][i]),
                         dq_nom=float(np.linalg.norm(qt[1:] - q_nom[1:])), n_cams=float(batch["cam_present"][i].sum()),
                         lidar_valid=float(batch["lidar_valid"][i].mean()),
                         **{f"rel_{j}": float(batch["rel_gt"][i][j]) for j in range(4)})
                rows.append(m)
                if n_saved < args.save_preds and suite in ("suite_clean", "suite_all_s2", "suite_ood_artic"):
                    os.makedirs(preds_dir, exist_ok=True)
                    np.savez_compressed(os.path.join(preds_dir, f"{suite}_{n_saved}.npz"),
                                        **{k: v for k, v in pi.items() if v is not None},
                                        **{f"gt_{k}": v for k, v in gt.items() if not np.isscalar(v)},
                                        q_true=qt, scene_id=int(batch["scene_id"][i]), frame_id=int(batch["frame_id"][i]))
                    n_saved += 1
        print(f"[eval] {suite}: {len(ds)} frames", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    with open(os.path.join(args.out, f"{args.tag}_s{args.seed}.json"), "w") as f:
        json.dump(dict(tag=args.tag, variant=variant, seed=args.seed, runs=args.run, conformal=conf,
                       ms_per_frame=1000 * float(np.median(lat)) if lat else None, params=count_params(models[0]),
                       gpu_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else None,
                       lam=args.lam), f, indent=1)
    print(f"[eval] wrote {csv_path} ({len(df)} rows), {1000 * np.median(lat):.1f} ms/frame")


if __name__ == "__main__":
    main()
