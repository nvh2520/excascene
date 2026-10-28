"""Real-image module (optional, ~2 GPU-hours).

Trains the same 4-channel ResNet-18/FPN encoder used by KinAfford-Ex for semantic segmentation on GOOSE
(depth channel = 0), then evaluates mIoU / ECE / NLL / error-detection AUROC on GOOSE-Ex frames (per
platform, e.g. the excavator) under the SAME photometric degradations as the synthetic benchmark.
Exports the encoder so that the `R2_full_gooseinit` variant can be trained from real-image weights.

    python -m kinafford.real.goose_seg --manifest data_real/manifest.json --out runs_real/goose_R2 \
        [--epochs 6 --batch 16 --eval_platforms excavator spot --num_classes 64]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from ..models.blocks import ImageEncoder
from ..sim.degrade import degrade_camera
from ..metrics import ece_bins, ece_from_bins, auroc
from ..data.dataset import IMAGENET_MEAN, IMAGENET_STD

IGNORE = 255


class RealSegDataset(Dataset):
    def __init__(self, entries, size=(384, 240), crop=(320, 192), train=False, num_classes=64, degrade=None, seed=0):
        self.entries, self.size, self.crop, self.train = entries, size, crop, train
        self.num_classes, self.degrade = num_classes, degrade
        self.seed = seed

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = Image.open(e["rgb"]).convert("RGB").resize(self.size, Image.BILINEAR)
        lab = Image.open(e["label"]).resize(self.size, Image.NEAREST)
        img = np.asarray(img, dtype=np.uint8)
        lab = np.asarray(lab).astype(np.int64)
        if lab.ndim == 3:
            lab = lab[..., 0]
        lab[lab >= self.num_classes] = IGNORE
        if self.train:
            W, H = self.size
            cw, ch = self.crop
            x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
            img, lab = img[y0:y0 + ch, x0:x0 + cw], lab[y0:y0 + ch, x0:x0 + cw]
            if random.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            if random.random() < 0.5:  # light photometric augmentation (train severities 1-2 only)
                rng = np.random.default_rng(random.randint(0, 2 ** 31))
                kind = str(rng.choice(["dust", "blur", "noise", "occlusion"]))
                img, _ = degrade_camera(np.ascontiguousarray(img), np.ones(img.shape[:2], np.float32), kind, int(rng.integers(1, 3)), rng)
        elif self.degrade is not None:
            kind, level = self.degrade
            rng = np.random.default_rng(self.seed * 100003 + i)
            img, _ = degrade_camera(np.ascontiguousarray(img), np.ones(img.shape[:2], np.float32), kind, level, rng)
        x = (np.ascontiguousarray(img).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = np.concatenate([x.transpose(2, 0, 1), np.zeros((1,) + x.shape[:2], np.float32)], 0)  # depth channel = 0
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(lab))


class SegNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.encoder = ImageEncoder(64, pretrained=pretrained)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, num_classes, 1))

    def forward(self, x):
        p, _, _ = self.encoder(x)
        return F.interpolate(self.head(p), size=x.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    csum = asum = cnt = None
    nll, n_pix, ent_all, wrong_all = 0.0, 0, [], []
    rng = np.random.default_rng(0)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        logp = F.log_softmax(logits.float(), 1)
        prob = logp.exp()
        pred = prob.argmax(1)
        valid = y != IGNORE
        yv, pv = y[valid], pred[valid]
        cm += np.bincount((yv * num_classes + pv).cpu().numpy(), minlength=num_classes ** 2).reshape(num_classes, num_classes)
        nll += -logp.gather(1, torch.where(valid, y, 0)[:, None])[:, 0][valid].sum().item()
        n_pix += int(valid.sum())
        conf = prob.max(1).values[valid].cpu().numpy()
        correct = (pv == yv).cpu().numpy()
        c, a, k = ece_bins(conf, correct)
        csum = c if csum is None else csum + c
        asum = a if asum is None else asum + a
        cnt = k if cnt is None else cnt + k
        ent = -(prob * logp).sum(1)[valid].cpu().numpy()
        sel = rng.choice(len(ent), size=min(4000, len(ent)), replace=False)
        ent_all.append(ent[sel])
        wrong_all.append((~correct)[sel].astype(int))
    present = cm.sum(1) > 0
    iou = np.diag(cm) / np.maximum(cm.sum(1) + cm.sum(0) - np.diag(cm), 1)
    model.train()
    return dict(miou=float(iou[present].mean()), pixel_acc=float(np.diag(cm).sum() / max(cm.sum(), 1)),
                ece=ece_from_bins(csum, asum, cnt), nll=nll / max(n_pix, 1),
                err_auroc=auroc(np.concatenate(ent_all), np.concatenate(wrong_all)), n_classes_present=int(present.sum()),
                n_images=len(loader.dataset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_classes", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--eval_platforms", nargs="*", default=["excavator"])
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--eval_limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    entries = json.load(open(args.manifest))
    train_e = [e for e in entries if e["dataset"] == "goose" and e["split"] == "train"][: args.train_limit]
    val_e = [e for e in entries if e["dataset"] == "goose" and e["split"] == "val"][: args.eval_limit]
    ex_e = {p: [e for e in entries if e["dataset"] == "goose_ex" and e["platform"] == p][: args.eval_limit] for p in args.eval_platforms}
    print(f"[real] train={len(train_e)} val={len(val_e)} goose_ex={{ {', '.join(f'{k}:{len(v)}' for k, v in ex_e.items())} }}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNet(args.num_classes).to(device)
    ckpt = os.path.join(args.out, "segnet.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print("[real] loaded existing checkpoint; skipping training")
    else:
        dl = DataLoader(RealSegDataset(train_e, train=True, num_classes=args.num_classes), batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
        t0, step = time.time(), 0
        for ep in range(args.epochs):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = F.cross_entropy(model(x).float(), y, ignore_index=IGNORE)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 50 == 0:
                    print(f"[real] ep {ep} step {step} loss {loss.item():.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            v = evaluate(model, DataLoader(RealSegDataset(val_e, num_classes=args.num_classes), batch_size=args.batch,
                                           num_workers=args.num_workers), device, args.num_classes)
            print(f"[real] epoch {ep} val: {v}", flush=True)
        torch.save(model.state_dict(), ckpt)
    torch.save(dict(encoder_state=model.encoder.state_dict()), os.path.join(args.out, "encoder.pt"))
    # robustness / calibration evaluation on GOOSE val and GOOSE-Ex platforms
    rows = []
    conds = [("clean", 0)] + [(k, l) for k in ("dust", "blur", "noise", "occlusion") for l in (1, 2, 3)]
    for name, es in [("goose_val", val_e)] + [(f"goose_ex_{p}", v) for p, v in ex_e.items()]:
        if not es:
            continue
        for kind, lvl in conds:
            ds = RealSegDataset(es, num_classes=args.num_classes, degrade=None if lvl == 0 else (kind, lvl), seed=args.seed)
            r = evaluate(model, DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers), device, args.num_classes)
            r.update(eval_set=name, degradation=kind, level=lvl)
            rows.append(r)
            print(f"[real] {name} {kind} L{lvl}: mIoU={r['miou']:.3f} ECE={r['ece']:.3f} NLL={r['nll']:.3f} AUROC={r['err_auroc']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "real_results.csv"), index=False)
    print(f"[real] wrote {os.path.join(args.out, 'real_results.csv')} and encoder.pt")


if __name__ == "__main__":
    main()
