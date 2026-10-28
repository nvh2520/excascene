"""Sensor-degradation models with three severity levels (1 = mild, 2 = moderate, 3 = severe).

Training frames see levels {1, 2} at random; level 3 is held out (extrapolation).
Geometric faults (extrinsic drift, joint-encoder error) are baked into the rendering because they
change the *true* sensor pose while the model keeps using the *nominal* pose / *reported* joints.
Photometric and point-level faults are applied to the rendered data.  All sampling is seeded.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .kinematics import axis_angle, trans, CAMERAS, SENSOR_ORDER
from .scene import SEM_SKY

# ------------------------------------------------------------------------- severity tables
SEV = {
    "occl_frac": [0.10, 0.25, 0.45],
    "dust_alpha": [0.25, 0.45, 0.65],
    "dust_holes": [0.10, 0.25, 0.45],
    "blur_sigma": [1.0, 2.5, 4.5],
    "exposure": [(0.70, 1.40), (0.50, 1.70), (0.35, 2.20)],
    "noise_sigma": [6.0, 14.0, 26.0],
    "depth_rel": [0.01, 0.03, 0.07],
    "depth_holes": [0.05, 0.15, 0.35],
    "lidar_drop": [0.20, 0.50, 0.80],
    "lidar_ring_keep": [1, 2, 4],
    "lidar_range_sigma": [0.02, 0.06, 0.15],
    "lidar_spurious": [0.01, 0.04, 0.10],
    "ext_rot_deg": [0.5, 2.0, 5.0],
    "ext_trans_m": [0.02, 0.06, 0.15],
    "joint_bias_deg": [0.5, 2.0, 5.0],
}
CAM_PHOTO_KINDS = ["occlusion", "dust", "blur", "noise", "depth"]

# suites: name -> spec.  'photo'/'lidar'/'ext'/'joint' hold severity levels (0 = none); 'drop' camera dropout
SUITES = {
    "clean": dict(photo=0, lidar=0, ext=0, joint=0, drop=0.0, artic=False, terrain=False),
    "iid": dict(train_mix=True, artic=False, terrain=False),
    "cam_s1": dict(photo=1, lidar=0, ext=0, joint=0, drop=0.0),
    "cam_s2": dict(photo=2, lidar=0, ext=0, joint=0, drop=0.0),
    "cam_s3": dict(photo=3, lidar=0, ext=0, joint=0, drop=1.0),   # drop = 1 -> one random camera dropped
    "lidar_s1": dict(photo=0, lidar=1, ext=0, joint=0, drop=0.0),
    "lidar_s2": dict(photo=0, lidar=2, ext=0, joint=0, drop=0.0),
    "lidar_s3": dict(photo=0, lidar=3, ext=0, joint=0, drop=0.0),
    "ext_s1": dict(photo=0, lidar=0, ext=1, joint=0, drop=0.0),
    "ext_s2": dict(photo=0, lidar=0, ext=2, joint=0, drop=0.0),
    "ext_s3": dict(photo=0, lidar=0, ext=3, joint=0, drop=0.0),
    "joint_s1": dict(photo=0, lidar=0, ext=0, joint=1, drop=0.0),
    "joint_s2": dict(photo=0, lidar=0, ext=0, joint=2, drop=0.0),
    "joint_s3": dict(photo=0, lidar=0, ext=0, joint=3, drop=0.0),
    "all_s2": dict(photo=2, lidar=2, ext=2, joint=2, drop=0.0),
    "all_s3": dict(photo=3, lidar=3, ext=3, joint=3, drop=1.0),
    "ood_artic": dict(train_mix=True, artic=True, terrain=False),
    "ood_terrain": dict(train_mix=True, artic=False, terrain=True),
}
SUITE_ORDER = list(SUITES.keys())


# ------------------------------------------------------------------------- camera degradations


def _ellipse_mask(H, W, rng, frac):
    mask = np.zeros((H, W), dtype=bool)
    yy, xx = np.mgrid[0:H, 0:W]
    target = frac * H * W
    while mask.sum() < target:
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        ry, rx = rng.uniform(8, 0.35 * H), rng.uniform(8, 0.35 * W)
        th = rng.uniform(0, np.pi)
        dx, dy = xx - cx, yy - cy
        lx = np.cos(th) * dx + np.sin(th) * dy
        ly = -np.sin(th) * dx + np.cos(th) * dy
        mask |= (lx / rx) ** 2 + (ly / ry) ** 2 <= 1
    return mask


def degrade_camera(rgb: np.ndarray, depth: np.ndarray, kind: str, level: int, rng: np.random.Generator):
    """rgb uint8 (H,W,3), depth float32 (H,W) (0 = invalid).  Returns new arrays."""
    if level <= 0:
        return rgb, depth
    i = level - 1
    H, W = depth.shape
    img = rgb.astype(np.float32)
    depth = depth.copy()
    if kind == "occlusion":
        m = _ellipse_mask(H, W, rng, SEV["occl_frac"][i])
        blob = rng.normal(0, 12, size=(H, W, 3)) + np.array([70, 55, 40])
        blob = gaussian_filter(blob, (3, 3, 0))
        img[m] = blob[m]
        depth[m] = 0.0
    elif kind == "dust":
        a = SEV["dust_alpha"][i]
        haze = np.array([200, 190, 170]) + gaussian_filter(rng.normal(0, 25, size=(H, W, 3)), (6, 6, 0))
        img = (1 - a) * img + a * haze
        holes = rng.uniform(size=(H, W)) < SEV["dust_holes"][i]
        depth[holes] = 0.0
    elif kind == "blur":
        img = gaussian_filter(img, (SEV["blur_sigma"][i], SEV["blur_sigma"][i], 0))
        lo, hi = SEV["exposure"][i]
        img = img * rng.uniform(lo, hi)
    elif kind == "noise":
        img = img + rng.normal(0, SEV["noise_sigma"][i], size=img.shape)
    elif kind == "depth":
        valid = depth > 0
        depth[valid] *= 1 + rng.normal(0, SEV["depth_rel"][i], size=valid.sum())
        holes = rng.uniform(size=(H, W)) < SEV["depth_holes"][i]
        depth[holes] = 0.0
    else:
        raise ValueError(kind)
    return np.clip(img, 0, 255).astype(np.uint8), depth.astype(np.float32)


def degrade_lidar(rng_img: np.ndarray, valid: np.ndarray, sem: np.ndarray, level: int, rng: np.random.Generator):
    if level <= 0:
        return rng_img, valid, sem
    i = level - 1
    rng_img, valid, sem = rng_img.astype(np.float32).copy(), valid.astype(bool).copy(), sem.copy()
    keep = SEV["lidar_ring_keep"][i]
    if keep > 1:
        valid[np.arange(valid.shape[0]) % keep != 0] = False
    drop = rng.uniform(size=valid.shape) < SEV["lidar_drop"][i]
    valid &= ~drop
    rng_img[valid] += rng.normal(0, SEV["lidar_range_sigma"][i], size=valid.sum())
    spur = (rng.uniform(size=valid.shape) < SEV["lidar_spurious"][i]) & ~drop
    rng_img[spur] = rng.uniform(0.5, 2.5, size=spur.sum())
    valid[spur] = True
    sem[spur] = SEM_SKY  # "unknown" return
    rng_img[~valid] = 0.0
    return rng_img.astype(np.float32), valid, sem


def drift_transform(level: int, rng: np.random.Generator) -> np.ndarray:
    if level <= 0:
        return np.eye(4)
    i = level - 1
    axis = rng.normal(size=3)
    ang = np.deg2rad(SEV["ext_rot_deg"][i]) * rng.uniform(0.7, 1.0)
    t = rng.normal(size=3)
    t = t / (np.linalg.norm(t) + 1e-9) * SEV["ext_trans_m"][i] * rng.uniform(0.7, 1.0)
    return trans(t) @ axis_angle(axis, ang)


def joint_bias(level: int, rng: np.random.Generator) -> np.ndarray:
    """Additive error of the *reported* boom/arm/bucket angles (rad); swing is assumed exact."""
    b = np.zeros(4)
    if level <= 0:
        return b
    i = level - 1
    mag = np.deg2rad(SEV["joint_bias_deg"][i])
    b[1:] = rng.choice([-1.0, 1.0], size=3) * mag * rng.uniform(0.6, 1.0, size=3)
    return b


# ------------------------------------------------------------------------- per-frame specification


def sample_spec(suite: str, rng: np.random.Generator) -> dict:
    """Returns the degradation specification for one frame.
    spec = dict(cam={name: dict(drop, photo=[(kind, level), ...], ext)}, lidar=dict(level, ext), joint=level)"""
    s = SUITES[suite]
    spec = dict(cam={}, lidar=dict(level=0, ext=0), joint=0, suite=suite)
    if s.get("train_mix", False):
        for c in CAMERAS:
            photo = []
            if rng.uniform() < 0.35:
                kinds = rng.choice(CAM_PHOTO_KINDS, size=rng.integers(1, 3), replace=False)
                photo = [(str(k), int(rng.integers(1, 3))) for k in kinds]
            spec["cam"][c] = dict(drop=bool(rng.uniform() < 0.08), photo=photo,
                                  ext=int(rng.integers(1, 3)) if rng.uniform() < 0.25 else 0)
        spec["lidar"] = dict(level=int(rng.integers(1, 3)) if rng.uniform() < 0.30 else 0,
                             ext=int(rng.integers(1, 3)) if rng.uniform() < 0.25 else 0)
        spec["joint"] = int(rng.integers(1, 3)) if rng.uniform() < 0.25 else 0
        return spec
    dropped = rng.choice(CAMERAS) if s.get("drop", 0.0) > 0 else None
    for c in CAMERAS:
        photo = [(k, s["photo"]) for k in CAM_PHOTO_KINDS] if s["photo"] > 0 else []
        spec["cam"][c] = dict(drop=(c == dropped), photo=photo, ext=s["ext"])
    spec["lidar"] = dict(level=s["lidar"], ext=s["ext"])
    spec["joint"] = s["joint"]
    return spec


def reliability_targets(spec: dict) -> np.ndarray:
    """Per-sensor reliability in [0,1] (order: cab, boom, stick, lidar) = 1 - max severity / 3, 0 if dropped."""
    rel = np.ones(4, dtype=np.float32)
    for k, c in enumerate(CAMERAS):
        cs = spec["cam"][c]
        if cs["drop"]:
            rel[k] = 0.0
            continue
        sev = max([lvl for _, lvl in cs["photo"]] + [cs["ext"], spec["joint"] if c != "cab" else 0] + [0])
        rel[k] = 1.0 - sev / 3.0
    sev = max(spec["lidar"]["level"], spec["lidar"]["ext"])
    rel[3] = 1.0 - sev / 3.0
    return rel
