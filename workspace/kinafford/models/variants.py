"""Model variants.

cfg keys (see configs/variants.yaml):
  sensors     : subset of [cab, boom, stick, lidar]
  extrinsics  : 'dynamic' (FK from reported joints) | 'static' (nominal articulation for boom/stick cams)
  fusion      : 'geometric' | 'late' | 'perceiver'
  gating      : bool  (learned per-sensor reliability gates)
  hetero      : bool  (heteroscedastic uncertainty heads)
  aff_head    : bool  (direct decision-centric affordance head)
  pretrained  : bool  (ImageNet init of the image encoder)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sim.kinematics import SENSOR_ORDER, CAMERAS, SENSORS, lidar_rays
from ..sim.scene import BEV_N
from .blocks import (ImageEncoder, CameraSplat, LidarPillars, BEVUNet, Heads, SensorGate, sincos_pos, FEAT_H, FEAT_W)

SENSOR_IDX = {n: i for i, n in enumerate(SENSOR_ORDER)}


class GeometricFusionNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.sensors = list(cfg["sensors"])
        self.cams = [s for s in self.sensors if s in CAMERAS]
        self.use_lidar = "lidar" in self.sensors
        C = cfg.get("feat_ch", 64)
        self.encoder = ImageEncoder(C, pretrained=cfg.get("pretrained", True)) if self.cams else None
        self.splat = CameraSplat()
        if self.use_lidar:
            self.pillars = LidarPillars(lidar_rays(SENSORS["lidar"]), C)
        self.gating = cfg.get("gating", False)
        n_sens = len(self.sensors)
        if self.gating:
            self.gates = nn.ModuleDict({s: SensorGate(C + 3) for s in self.sensors})
        self.late = cfg.get("fusion", "geometric") == "late"
        in_ch = (C + 3) if self.late else n_sens * (C + 3) + (n_sens if self.gating else 0)
        self.inproj = nn.Conv2d(in_ch, 64, 1)
        if self.late:
            self.sensor_emb = nn.Embedding(4, 4)
        self.unet = BEVUNet(64, base=64, cond_dim=4 + (4 if self.late else 0))
        self.heads = Heads(self.unet.out_ch, cfg.get("hetero", False), cfg.get("aff_head", False))

    def sensor_maps(self, batch):
        T_all = batch["T_dyn"] if self.cfg.get("extrinsics", "dynamic") == "dynamic" else batch["T_static"]
        maps, seg_logits, present = {}, [], {}
        if self.cams:
            B = batch["img"].shape[0]
            ci = [CAMERAS.index(c) for c in self.cams]
            x = batch["img"][:, ci].reshape(-1, 4, batch["img"].shape[-2], batch["img"].shape[-1])
            p, _, seg = self.encoder(x)
            p = p.view(B, len(ci), *p.shape[1:])
            seg = seg.view(B, len(ci), *seg.shape[1:])
            for j, c in enumerate(self.cams):
                k = CAMERAS.index(c)
                maps[c] = self.splat(p[:, j], batch["depth"][:, k], batch["K"][:, k], T_all[:, SENSOR_IDX[c]], batch["cam_present"][:, k])
                present[c] = batch["cam_present"][:, k]
                seg_logits.append((k, seg[:, j]))
        if self.use_lidar:
            maps["lidar"] = self.pillars(batch["lidar_range"], batch["lidar_valid"], T_all[:, SENSOR_IDX["lidar"]])
            present["lidar"] = torch.ones_like(batch["meta"][:, 0])
        return maps, seg_logits, present

    def forward(self, batch):
        maps, seg_logits, present = self.sensor_maps(batch)
        q = batch["q"]
        out = {"img_seg": seg_logits}
        gate_scalars = {}
        if self.gating:
            gated = {}
            gmaps = []
            for s in self.sensors:
                g, gs = self.gates[s](maps[s], batch["meta"], q)
                g = g * present[s][:, None, None, None]
                gated[s] = maps[s] * g
                gmaps.append(g)
                gate_scalars[s] = gs * present[s]
            maps = gated
        if self.late:
            outs = []
            for s in self.sensors:
                emb = self.sensor_emb.weight[SENSOR_IDX[s]][None].expand(q.shape[0], -1)
                f = self.unet(self.inproj(maps[s]), torch.cat([q, emb], 1))
                outs.append((self.heads(f), present[s]))
            wsum = sum(p for _, p in outs).clamp(min=1.0)
            merged = {}
            for k in outs[0][0]:
                merged[k] = sum(o[k] * p.view(-1, *([1] * (o[k].dim() - 1))) for o, p in outs) / wsum.view(-1, *([1] * (outs[0][0][k].dim() - 1)))
            out.update(merged)
        else:
            x = torch.cat([maps[s] for s in self.sensors] + (gmaps if self.gating else []), 1)
            f = self.unet(self.inproj(x), q)
            out.update(self.heads(f))
        if self.gating:
            out["gate"] = torch.stack([gate_scalars[s] for s in self.sensors], 1)  # (B, n_sensors)
        return out


class PerceiverNet(nn.Module):
    """Raw-token cross-attention fusion with NO explicit geometry (joint state given as a token)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        D = cfg.get("perceiver_dim", 192)
        self.encoder = ImageEncoder(64, pretrained=cfg.get("pretrained", True))
        self.img_proj = nn.Linear(256, D)
        h16, w16 = FEAT_H // 4, FEAT_W // 4  # stride-16 map 10 x 16
        self.register_buffer("img_pos", sincos_pos(h16, w16, D))
        self.cam_emb = nn.Embedding(3, D)
        self.lidar_cnn = nn.Sequential(
            nn.Conv2d(2, 32, (3, 5), stride=(2, 4), padding=(1, 2)), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, D, 3, stride=2, padding=1))
        self.register_buffer("lidar_pos", sincos_pos(4, 32, D))
        self.q_tok = nn.Sequential(nn.Linear(4, D), nn.ReLU(inplace=True), nn.Linear(D, D))
        self.meta_tok = nn.Sequential(nn.Linear(5, D), nn.ReLU(inplace=True), nn.Linear(D, D))
        self.nq = 24
        self.queries = nn.Parameter(torch.randn(self.nq * self.nq, D) * 0.02)
        self.register_buffer("q_pos", sincos_pos(self.nq, self.nq, D))
        self.blocks = nn.ModuleList()
        for _ in range(cfg.get("perceiver_layers", 3)):
            self.blocks.append(nn.ModuleDict(dict(
                ca=nn.MultiheadAttention(D, 4, batch_first=True), n1=nn.LayerNorm(D),
                sa=nn.MultiheadAttention(D, 4, batch_first=True), n2=nn.LayerNorm(D),
                ff=nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)), n3=nn.LayerNorm(D))))
        self.up = nn.Sequential(
            nn.Conv2d(D, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 96, 3, padding=1), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.heads = Heads(64, cfg.get("hetero", False), cfg.get("aff_head", False))

    def forward(self, batch):
        B = batch["img"].shape[0]
        x = batch["img"].reshape(-1, 4, batch["img"].shape[-2], batch["img"].shape[-1])
        _, f3, seg = self.encoder(x)
        f3 = f3.view(B, 3, f3.shape[1], -1).permute(0, 1, 3, 2)                     # (B,3,160,256)
        tok = self.img_proj(f3) + self.img_pos[None, None] + self.cam_emb.weight[None, :, None]
        tok = tok.reshape(B, -1, tok.shape[-1])
        img_mask = (batch["cam_present"] < 0.5)[:, :, None].expand(-1, -1, f3.shape[2]).reshape(B, -1)
        lid = self.lidar_cnn(torch.stack([batch["lidar_range"] / 40.0, batch["lidar_valid"]], 1))  # (B,D,4,32)
        lid = lid.flatten(2).transpose(1, 2) + self.lidar_pos[None]
        extra = torch.stack([self.q_tok(batch["q"]), self.meta_tok(batch["meta"])], 1)
        tokens = torch.cat([tok, lid, extra], 1)
        mask = torch.cat([img_mask, torch.zeros(B, lid.shape[1] + 2, dtype=torch.bool, device=tok.device)], 1)
        z = (self.queries + self.q_pos)[None].expand(B, -1, -1)
        for blk in self.blocks:
            z = z + blk["ca"](blk["n1"](z), tokens, tokens, key_padding_mask=mask, need_weights=False)[0]
            z = z + blk["sa"](blk["n2"](z), blk["n2"](z), blk["n2"](z), need_weights=False)[0]
            z = z + blk["ff"](blk["n3"](z))
        f = z.transpose(1, 2).reshape(B, -1, self.nq, self.nq)
        f = self.up(f)
        out = self.heads(f)
        out["img_seg"] = [(k, seg.view(B, 3, *seg.shape[1:])[:, k]) for k in range(3)]
        return out


def build_model(cfg: dict) -> nn.Module:
    if cfg.get("fusion", "geometric") == "perceiver":
        return PerceiverNet(cfg)
    return GeometricFusionNet(cfg)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
