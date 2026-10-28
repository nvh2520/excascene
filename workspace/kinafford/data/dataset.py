"""ExcaScene dataset -> model inputs.

Extrinsics given to the model are computed from *nominal* mounts and *reported* joints (never from the
true drifted poses); the STATIC baseline uses the nominal articulation q_nom for boom/stick cameras.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..sim.kinematics import (SENSOR_ORDER, CAMERAS, SENSORS, EXCAVATOR, sensor_pose_U, camera_intrinsics,
                              lidar_rays)
from ..sim.scene import N_BEV_SEM

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEPTH_SCALE = 20.0
RANGE_SCALE = 40.0


def static_q(q_rep):
    q = np.asarray(q_rep, dtype=np.float64).copy()
    q[1:] = EXCAVATOR["q_nom"][1:]
    return q


def extrinsics(q_rep, kin_jitter_rad=0.0, rng=None):
    """(T_dyn, T_static) each (4,4,4) float32 for [cab, boom, stick, lidar]."""
    q = np.asarray(q_rep, dtype=np.float64).copy()
    if kin_jitter_rad > 0 and rng is not None:
        q[1:] += rng.normal(0, kin_jitter_rad, size=3)
    qs = static_q(q)
    T_dyn = np.stack([sensor_pose_U(n, q) for n in SENSOR_ORDER]).astype(np.float32)
    T_sta = np.stack([sensor_pose_U(n, qs) for n in SENSOR_ORDER]).astype(np.float32)
    return T_dyn, T_sta


class ExcaSceneDataset(Dataset):
    def __init__(self, root: str, split: str, train: bool = False, kin_jitter_deg: float = 0.0,
                 color_jitter: float = 0.0, limit: int | None = None):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npz")))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"no frames under {root}/{split}")
        self.train = train
        self.kin_jitter = np.deg2rad(kin_jitter_deg)
        self.color_jitter = color_jitter
        self.K = np.stack([camera_intrinsics(SENSORS[c]) for c in CAMERAS]).astype(np.float32)
        self.lidar_dirs = lidar_rays(SENSORS["lidar"]).astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        rng = np.random.default_rng((i * 7919 + int(d["scene_id"]) * 31 + int(d["frame_id"])) if self.train else 0)
        rgb = d["rgb"].astype(np.float32) / 255.0                      # (3,H,W,3)
        if self.train and self.color_jitter > 0:
            g = rng.uniform(1 - self.color_jitter, 1 + self.color_jitter, size=(3, 1, 1, 1))
            b = rng.uniform(-self.color_jitter, self.color_jitter, size=(3, 1, 1, 1)) * 0.3
            rgb = np.clip(rgb * g + b, 0, 1)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        depth = d["depth"].astype(np.float32)                          # (3,H,W)
        img = np.concatenate([rgb.transpose(0, 3, 1, 2), (depth / DEPTH_SCALE)[:, None]], axis=1)  # (3,4,H,W)
        present = d["cam_present"].astype(np.float32)
        img = img * present[:, None, None, None]
        lidar_range = d["lidar_range"].astype(np.float32)
        lidar_valid = d["lidar_valid"].astype(np.float32)
        q_rep = d["q_rep"].astype(np.float32)
        T_dyn, T_sta = extrinsics(q_rep, self.kin_jitter if self.train else 0.0, rng)
        meta = np.concatenate([present, [lidar_valid.mean()], [1.0]]).astype(np.float32)
        sample = dict(
            img=torch.from_numpy(img.astype(np.float32)), depth=torch.from_numpy(depth), img_sem=torch.from_numpy(d["img_sem"].astype(np.int64)),
            cam_present=torch.from_numpy(present),
            lidar_range=torch.from_numpy(lidar_range), lidar_valid=torch.from_numpy(lidar_valid),
            q=torch.from_numpy(q_rep), q_true=torch.from_numpy(d["q_true"].astype(np.float32)),
            T_dyn=torch.from_numpy(T_dyn), T_static=torch.from_numpy(T_sta), K=torch.from_numpy(self.K),
            meta=torch.from_numpy(meta),
            gt_H=torch.from_numpy(d["gt_H"].astype(np.float32)),
            gt_sem=torch.from_numpy(d["gt_sem"].astype(np.int64)),
            gt_mat=torch.from_numpy(d["gt_mat"].astype(np.int64)),
            gt_haz=torch.from_numpy(d["gt_haz"].astype(np.float32)),
            gt_aff=torch.from_numpy(d["gt_aff"].astype(np.float32)),
            gt_reach=torch.from_numpy(d["gt_reach"].astype(np.float32)),
            gt_vis=torch.from_numpy(d["gt_vis"].astype(np.float32)),
            rel_gt=torch.from_numpy(d["rel_gt"].astype(np.float32)),
            z_target=torch.tensor(float(d["z_target"])),
            scene_id=torch.tensor(int(d["scene_id"])), frame_id=torch.tensor(int(d["frame_id"])),
        )
        return sample


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
