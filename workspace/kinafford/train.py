"""Train one model variant.

    python -m kinafford.train --variant R2_full --seed 0 --out runs/R2_full/seed0 \
        --config configs/default.yaml --variants configs/variants.yaml [--set epochs=1 --set train_limit=64]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import ExcaSceneDataset, to_device
from .models.variants import build_model, count_params
from .losses import compute_losses
from .sim.scene import SEM_TERRAIN


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_config(args) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.variants) as f:
        variants = yaml.safe_load(f)
    if args.variant not in variants:
        raise KeyError(f"unknown variant {args.variant}; known: {list(variants)}")
    model_cfg = dict(variants["_base"])
    model_cfg.update(variants[args.variant])
    cfg["model"] = model_cfg
    cfg["variant"] = args.variant
    cfg["seed"] = args.seed
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        if k.startswith("model."):
            cfg["model"][k[6:]] = v
        else:
            cfg[k] = v
    return cfg


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    agg = dict(n=0, loss=0.0, H_mae_vis=0.0, aff_mae=0.0, sem_acc=0.0)
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"] and device.type == "cuda"):
            out = model(batch)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        losses = compute_losses(out, batch, cfg["model"])
        B = batch["gt_H"].shape[0]
        vis = batch["gt_vis"]
        agg["loss"] += losses["total"].item() * B
        agg["H_mae_vis"] += (((out["H"] - batch["gt_H"]).abs() * vis).sum() / vis.sum().clamp(min=1)).item() * B
        if "aff" in out:
            reach = (batch["gt_reach"] > 0.05).float()
            agg["aff_mae"] += (((torch.sigmoid(out["aff"]) - batch["gt_aff"]).abs() * reach).sum() / reach.sum().clamp(min=1)).item() * B
        agg["sem_acc"] += ((out["sem"].argmax(1) == batch["gt_sem"]).float() * vis).sum().item() / vis.sum().clamp(min=1).item() * B
        agg["n"] += B
    model.train()
    n = max(agg.pop("n"), 1)
    return {k: v / n for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--variants", default="configs/variants.yaml")
    ap.add_argument("--set", action="append", help="override, e.g. --set epochs=1 or --set model.gating=false")
    args = ap.parse_args()
    cfg = load_config(args)
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "done.json")):
        print(f"[train] {args.out} already done, skipping")
        return
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(12_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark = True

    mcfg = cfg["model"]
    train_ds = ExcaSceneDataset(cfg["data_root"], "train", train=True, kin_jitter_deg=mcfg.get("kin_jitter_deg", 0.0),
                                color_jitter=cfg.get("color_jitter", 0.0), limit=cfg.get("train_limit"))
    val_ds = ExcaSceneDataset(cfg["data_root"], "val", train=False, limit=cfg.get("val_limit"))
    nw = int(cfg.get("num_workers", 8))
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, drop_last=True,
                          pin_memory=True, persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=max(nw // 2, 0), pin_memory=True)

    model = build_model(mcfg).to(device)
    init_from = mcfg.get("init_from")
    if init_from:
        sd = torch.load(init_from, map_location="cpu")
        sd = sd.get("encoder_state", sd)
        missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
        print(f"[train] initialised encoder from {init_from}: missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[train] variant={args.variant} seed={args.seed} params={count_params(model) / 1e6:.2f}M train={len(train_ds)} val={len(val_ds)}")

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    other = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": other, "lr": cfg["lr"]}, {"params": enc_params, "lr": cfg["lr_encoder"]}],
                            weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(train_dl)
    warm = min(cfg.get("warmup_steps", 300), max(total_steps // 10, 1))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * p)) * 0.98 + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best, step, t0 = float("inf"), 0, time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        run = {}
        for batch in train_dl:
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"] and device.type == "cuda"):
                out = model(batch)
            out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
            losses = compute_losses(out, batch, mcfg)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + v.item()
            if step % 50 == 0:
                msg = {k: round(v / 50, 4) for k, v in run.items()}
                msg.update(step=step, epoch=epoch, lr=sched.get_last_lr()[0], min=round((time.time() - t0) / 60, 1))
                print("[train]", json.dumps(msg), flush=True)
                log.write(json.dumps(msg) + "\n")
                log.flush()
                run = {}
        val = validate(model, val_dl, device, cfg)
        val.update(epoch=epoch, step=step, min=round((time.time() - t0) / 60, 1))
        print("[val]", json.dumps(val), flush=True)
        log.write(json.dumps(dict(val=val)) + "\n")
        log.flush()
        score = val["H_mae_vis"] + val.get("aff_mae", 0.0)
        state = dict(model=model.state_dict(), cfg=cfg, epoch=epoch, val=val)
        torch.save(state, os.path.join(args.out, "last.pt"))
        if score < best:
            best = score
            torch.save(state, os.path.join(args.out, "best.pt"))
    with open(os.path.join(args.out, "done.json"), "w") as f:
        json.dump(dict(best_score=best, minutes=(time.time() - t0) / 60, params=count_params(model), steps=step,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else 0), f, indent=1)
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
