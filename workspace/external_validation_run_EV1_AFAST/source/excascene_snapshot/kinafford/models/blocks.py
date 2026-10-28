"""Building blocks shared by all model variants."""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sim.scene import BEV_N, BEV_RES, BEV_X0, BEV_Y0, N_BEV_SEM, MAT_CLASSES
from ..sim.kinematics import IMG_H, IMG_W

STRIDE = 4
FEAT_H, FEAT_W = IMG_H // STRIDE, IMG_W // STRIDE   # 40 x 64
N_IMG_SEM = 7


# --------------------------------------------------------------------------- image encoder


class ImageEncoder(nn.Module):
    """ResNet-18 with a 4-channel (RGB + depth) stem and an FPN-lite that yields stride-4 features."""

    def __init__(self, out_ch: int = 64, pretrained: bool = True):
        super().__init__()
        import torchvision
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            except Exception:  # pragma: no cover
                warnings.warn("torchvision weights enum unavailable; random init")
        try:
            r = torchvision.models.resnet18(weights=weights)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"could not load pretrained ResNet-18 ({e}); using random init")
            r = torchvision.models.resnet18(weights=None)
        conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = r.conv1.weight
            conv1.weight[:, 3:] = r.conv1.weight.mean(1, keepdim=True)
        self.stem = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3 = r.layer1, r.layer2, r.layer3
        self.lat1, self.lat2, self.lat3 = nn.Conv2d(64, out_ch, 1), nn.Conv2d(128, out_ch, 1), nn.Conv2d(256, out_ch, 1)
        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.seg = nn.Conv2d(out_ch, N_IMG_SEM, 1)
        self.out_ch = out_ch

    def forward(self, x):
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        p = self.lat1(f1)
        p = p + F.interpolate(self.lat2(f2), size=p.shape[-2:], mode="bilinear", align_corners=False)
        p = p + F.interpolate(self.lat3(f3), size=p.shape[-2:], mode="bilinear", align_corners=False)
        p = F.relu(self.smooth(p))
        return p, f3, self.seg(p)


# --------------------------------------------------------------------------- BEV projection


def bev_cell_index(px, py):
    ix = torch.floor((px - BEV_X0) / BEV_RES).long()
    iy = torch.floor((py - BEV_Y0) / BEV_RES).long()
    ok = (ix >= 0) & (ix < BEV_N) & (iy >= 0) & (iy < BEV_N)
    return ix, iy, ok


def scatter_bev(feats, pz, idx, valid, B, C, reduce="mean"):
    """feats (M,C), pz (M,), idx (M,) flat cell index in [0, B*N*N), valid (M,) bool.
    Returns (B, C+3, N, N): reduced features, log1p(count), zmax, zmin."""
    NN = BEV_N * BEV_N
    idx = idx[valid]
    f = feats[valid].float()
    z = pz[valid].float()
    cnt = torch.zeros(B * NN, 1, device=feats.device).index_add_(0, idx, torch.ones_like(z)[:, None])
    if reduce == "mean":
        acc = torch.zeros(B * NN, C, device=feats.device).index_add_(0, idx, f)
        out = acc / cnt.clamp(min=1.0)
    else:
        out = torch.zeros(B * NN, C, device=feats.device)
        out.scatter_reduce_(0, idx[:, None].expand(-1, C), f, reduce="amax", include_self=False)
    zmax = torch.zeros(B * NN, 1, device=feats.device).scatter_reduce_(0, idx[:, None], z[:, None], reduce="amax", include_self=False)
    zmin = torch.zeros(B * NN, 1, device=feats.device).scatter_reduce_(0, idx[:, None], z[:, None], reduce="amin", include_self=False)
    has = cnt > 0
    zmax = torch.where(has, zmax, torch.zeros_like(zmax))
    zmin = torch.where(has, zmin, torch.zeros_like(zmin))
    res = torch.cat([out, torch.log1p(cnt), zmax, zmin], dim=1)  # (B*NN, C+3)
    return res.view(B, BEV_N, BEV_N, C + 3).permute(0, 3, 1, 2).contiguous()


class CameraSplat(nn.Module):
    """Unproject stride-4 image features with the measured depth and splat them into the U-frame BEV."""

    def forward(self, feats, depth, K, T, present):
        # feats (B,C,h,w) ; depth (B,H,W) metres ; K (B,3,3) ; T (B,4,4) = T_U_from_cam ; present (B,)
        with torch.autocast("cuda", enabled=False):  # geometry always in float32
            return self._forward(feats, depth.float(), K.float(), T.float(), present.float())

    def _forward(self, feats, depth, K, T, present):
        B, C, h, w = feats.shape
        d = depth[:, STRIDE // 2::STRIDE, STRIDE // 2::STRIDE][:, :h, :w]  # (B,h,w) centre sample
        u = (torch.arange(w, device=feats.device, dtype=torch.float32) * STRIDE + STRIDE / 2 + 0.5)
        v = (torch.arange(h, device=feats.device, dtype=torch.float32) * STRIDE + STRIDE / 2 + 0.5)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        fx, fy = K[:, 0, 0][:, None, None], K[:, 1, 1][:, None, None]
        cx, cy = K[:, 0, 2][:, None, None], K[:, 1, 2][:, None, None]
        x = (uu[None] - cx) / fx * d
        y = (vv[None] - cy) / fy * d
        P = torch.stack([x, y, d], dim=-1)                              # (B,h,w,3) camera frame
        R, t = T[:, :3, :3], T[:, :3, 3]
        PU = torch.einsum("bij,bhwj->bhwi", R, P) + t[:, None, None, :]
        ix, iy, ok = bev_cell_index(PU[..., 0], PU[..., 1])
        valid = ok & (d > 0.05) & (PU[..., 2] < 8.0) & (present[:, None, None] > 0.5)
        bidx = torch.arange(B, device=feats.device)[:, None, None].expand(B, h, w)
        idx = (bidx * BEV_N * BEV_N + iy.clamp(0, BEV_N - 1) * BEV_N + ix.clamp(0, BEV_N - 1)).reshape(-1)
        f = feats.permute(0, 2, 3, 1).reshape(-1, C)
        return scatter_bev(f, PU[..., 2].reshape(-1), idx, valid.reshape(-1), B, C, reduce="mean")


class LidarPillars(nn.Module):
    """PointPillars-lite: per-point MLP + max-pool into U-frame BEV cells."""

    def __init__(self, dirs, out_ch: int = 64):
        super().__init__()
        self.register_buffer("dirs", torch.as_tensor(dirs, dtype=torch.float32))  # (rings, az, 3)
        self.mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU(inplace=True), nn.Linear(32, out_ch), nn.ReLU(inplace=True))
        self.out_ch = out_ch
        rings = dirs.shape[0]
        self.register_buffer("ring_idx", torch.arange(rings, dtype=torch.float32)[:, None].expand(rings, dirs.shape[1]) / rings)

    def forward(self, rng, valid, T, max_range=40.0):
        # rng (B,rings,az) metres ; valid (B,rings,az) ; T (B,4,4) = T_U_from_lidar
        with torch.autocast("cuda", enabled=False):  # geometry always in float32
            return self._forward(rng.float(), valid.float(), T.float(), max_range)

    def _forward(self, rng, valid, T, max_range):
        B = rng.shape[0]
        P = rng[..., None] * self.dirs[None]                              # (B,r,a,3)
        R, t = T[:, :3, :3], T[:, :3, 3]
        PU = torch.einsum("bij,braj->brai", R, P) + t[:, None, None, :]
        ix, iy, ok = bev_cell_index(PU[..., 0], PU[..., 1])
        cxc = BEV_X0 + (ix.float() + 0.5) * BEV_RES
        cyc = BEV_Y0 + (iy.float() + 0.5) * BEV_RES
        feats = torch.stack([PU[..., 2], rng / max_range, PU[..., 0] - cxc, PU[..., 1] - cyc,
                             self.ring_idx[None].expand(B, -1, -1), valid], dim=-1)
        f = self.mlp(feats.reshape(-1, 6))
        v = (ok & (valid > 0.5) & (rng > 0.3) & (PU[..., 2] < 8.0)).reshape(-1)
        bidx = torch.arange(B, device=rng.device)[:, None, None].expand_as(ix)
        idx = (bidx * BEV_N * BEV_N + iy.clamp(0, BEV_N - 1) * BEV_N + ix.clamp(0, BEV_N - 1)).reshape(-1)
        return scatter_bev(f, PU[..., 2].reshape(-1), idx, v, B, self.out_ch, reduce="max")


# --------------------------------------------------------------------------- BEV decoder


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class BEVUNet(nn.Module):
    """96x96 UNet with FiLM conditioning on the joint state at the bottleneck."""

    def __init__(self, in_ch: int, base: int = 64, cond_dim: int = 4):
        super().__init__()
        self.e0 = conv_block(in_ch, base)
        self.e1 = conv_block(base, base * 2)
        self.e2 = conv_block(base * 2, base * 4)
        self.film = nn.Sequential(nn.Linear(cond_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, base * 8))
        self.u1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d1 = conv_block(base * 4, base * 2)
        self.u2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d2 = conv_block(base * 2, base)
        self.out_ch = base
        self.base = base

    def forward(self, x, cond):
        e0 = self.e0(x)
        e1 = self.e1(F.max_pool2d(e0, 2))
        e2 = self.e2(F.max_pool2d(e1, 2))
        gb = self.film(cond)
        gamma, beta = gb[:, :self.base * 4], gb[:, self.base * 4:]
        e2 = e2 * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        d1 = self.d1(torch.cat([self.u1(e2), e1], 1))
        d2 = self.d2(torch.cat([self.u2(d1), e0], 1))
        return d2


class Heads(nn.Module):
    def __init__(self, in_ch: int, hetero: bool, aff_head: bool):
        super().__init__()
        self.hetero, self.aff_head = hetero, aff_head
        self.H = nn.Conv2d(in_ch, 2 if hetero else 1, 1)
        self.sem = nn.Conv2d(in_ch, N_BEV_SEM, 1)
        self.mat = nn.Conv2d(in_ch, len(MAT_CLASSES), 1)
        self.haz = nn.Conv2d(in_ch, 1, 1)
        if aff_head:
            self.aff = nn.Conv2d(in_ch, 2 if hetero else 1, 1)

    def forward(self, f):
        h = self.H(f)
        out = dict(H=h[:, 0], sem=self.sem(f), mat=self.mat(f), haz=self.haz(f)[:, 0])
        if self.hetero:
            out["H_logvar"] = h[:, 1].clamp(-7, 5)
        if self.aff_head:
            a = self.aff(f)
            out["aff"] = a[:, 0]
            if self.hetero:
                out["aff_logvar"] = a[:, 1].clamp(-9, 3)
        return out


class SensorGate(nn.Module):
    """Learned reliability gate per sensor: spatial map + global scalar (from pooled features, meta, q)."""

    def __init__(self, ch: int, n_meta: int = 5):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        self.glob = nn.Sequential(nn.Linear(ch + n_meta + 4, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, S, meta, q):
        gs = self.spatial(S)
        gg = self.glob(torch.cat([S.mean((2, 3)), meta, q], 1))
        gate = torch.sigmoid(gs + gg[:, :, None, None])
        return gate, torch.sigmoid(gg)[:, 0]


def sincos_pos(h, w, dim):
    """2D sine-cosine positional embedding (h*w, dim)."""
    assert dim % 4 == 0
    d = dim // 4
    omega = torch.exp(-math.log(1e4) * torch.arange(d, dtype=torch.float32) / d)
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    py = ys.reshape(-1, 1) * omega[None]
    px = xs.reshape(-1, 1) * omega[None]
    return torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], dim=1)
