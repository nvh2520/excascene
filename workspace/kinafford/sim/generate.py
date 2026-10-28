"""Generate the ExcaScene benchmark (all splits + degradation suites).

    python -m kinafford.sim.generate --out data --profile full --workers 16
    python -m kinafford.sim.generate --out data_smoke --profile smoke --workers 4

Scene seeds are frozen per split (see PROFILES).  Every eval suite except ood_terrain re-uses the SAME
240 base scenes and joint configurations, so fault effects are paired across suites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from .kinematics import CAMERAS, SENSORS, SENSOR_ORDER
from .scene import generate_scene, sample_frame, bev_ground_truth
from .render import Renderer, visibility_mask
from .degrade import (SUITES, SUITE_ORDER, sample_spec, degrade_camera, degrade_lidar, drift_transform,
                      joint_bias, reliability_targets)

PROFILES = {
    "full": dict(train=(0, 3000, "iid"), val=(3000, 3300, "iid"), cal=(3300, 3600, "iid"),
                 test_iid=(3600, 4100, "iid"), suite_scenes=(5000, 5240), terrain_scenes=(6000, 6240), frames=4),
    "medium": dict(train=(0, 1500, "iid"), val=(3000, 3150, "iid"), cal=(3300, 3450, "iid"),
                   test_iid=(3600, 3850, "iid"), suite_scenes=(5000, 5120), terrain_scenes=(6000, 6120), frames=4),
    "smoke": dict(train=(0, 6, "iid"), val=(3000, 3002, "iid"), cal=(3300, 3302, "iid"),
                  test_iid=(3600, 3603, "iid"), suite_scenes=(5000, 5002), terrain_scenes=(6000, 6002), frames=2),
}


def frame_rng(suite: str, seed: int, k: int) -> np.random.Generator:
    h = hashlib.sha256(f"{suite}|{seed}|{k}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def render_frame(scene, renderer, seed, k, suite, ood_artic):
    q_true, phase = sample_frame(scene, k, ood_artic=ood_artic)
    rng = frame_rng(suite, seed, k)
    spec = sample_spec(suite, rng)
    q_rep = q_true + joint_bias(spec["joint"], rng)
    mb = renderer._machine_boxes(q_true)
    st = renderer._static_objects()

    rgb = np.zeros((3, SENSORS["cab"]["H"], SENSORS["cab"]["W"], 3), dtype=np.uint8)
    depth = np.zeros((3, SENSORS["cab"]["H"], SENSORS["cab"]["W"]), dtype=np.float16)
    img_sem = np.zeros((3, SENSORS["cab"]["H"], SENSORS["cab"]["W"]), dtype=np.uint8)
    present = np.zeros(3, dtype=np.uint8)
    drifts = np.stack([np.eye(4)] * 4).astype(np.float32)
    hits = []
    for i, c in enumerate(CAMERAS):
        cs = spec["cam"][c]
        if cs["drop"]:
            continue
        drift = drift_transform(cs["ext"], rng)
        drifts[i] = drift
        out = renderer.render_camera(c, q_true, rng, drift=drift, machine_boxes=mb, static=st)
        r, d = out["rgb"], out["depth"]
        for kind, lvl in cs["photo"]:
            r, d = degrade_camera(r, d, kind, lvl, rng)
        rgb[i], depth[i], img_sem[i], present[i] = r, d.astype(np.float16), out["sem"], 1
        hits.append(out["hits_B"])
    ldrift = drift_transform(spec["lidar"]["ext"], rng)
    drifts[3] = ldrift
    lo = renderer.render_lidar(q_true, rng, drift=ldrift, machine_boxes=mb, static=st)
    lrange, lvalid, lsem = degrade_lidar(lo["range"], lo["valid"], lo["sem"], spec["lidar"]["level"], rng)
    hits.append(lo["hits_B"])

    gt = bev_ground_truth(scene, q_true)
    vis = visibility_mask(hits, q_true)
    rel = reliability_targets(spec)
    return dict(
        rgb=rgb, depth=depth, img_sem=img_sem, cam_present=present,
        lidar_range=lrange.astype(np.float16), lidar_valid=lvalid.astype(np.uint8), lidar_sem=lsem,
        q_true=q_true.astype(np.float32), q_rep=q_rep.astype(np.float32), drift=drifts,
        gt_H=gt["H"].astype(np.float16), gt_sem=gt["sem"], gt_mat=gt["mat"], gt_haz=gt["haz"],
        gt_aff=gt["aff"].astype(np.float16), gt_reach=gt["reach"].astype(np.float16), gt_vis=vis,
        rel_gt=rel, z_target=np.float32(scene.z_target), scene_id=np.int32(seed), frame_id=np.int32(k),
        phase=np.str_(phase), spec=np.str_(json.dumps(spec)), suite=np.str_(suite),
    )


def _job(args):
    out_dir, seed, frames, suite, ood_artic, ood_terrain = args
    try:
        scene = generate_scene(seed, ood_terrain=ood_terrain)
        renderer = Renderer(scene)
        for k in range(frames):
            path = os.path.join(out_dir, f"f_{seed:06d}_{k}.npz")
            if os.path.exists(path):
                continue
            d = render_frame(scene, renderer, seed, k, suite, ood_artic)
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **d)
            os.replace(tmp, path)
        return (seed, None)
    except Exception as e:  # pragma: no cover
        import traceback
        return (seed, traceback.format_exc())


def build_jobs(out_root: str, profile: dict, splits=None):
    jobs = []
    def add(split_name, s0, s1, suite, artic=False, terrain=False):
        if splits and split_name not in splits:
            return
        out_dir = os.path.join(out_root, split_name)
        os.makedirs(out_dir, exist_ok=True)
        for seed in range(s0, s1):
            jobs.append((out_dir, seed, profile["frames"], suite, artic, terrain))
    for name in ("train", "val", "cal", "test_iid"):
        s0, s1, suite = profile[name]
        add(name, s0, s1, suite)
    a0, a1 = profile["suite_scenes"]
    t0, t1 = profile["terrain_scenes"]
    for suite in SUITE_ORDER:
        s = SUITES[suite]
        if s.get("terrain", False):
            add(f"suite_{suite}", t0, t1, suite, artic=False, terrain=True)
        else:
            add(f"suite_{suite}", a0, a1, suite, artic=s.get("artic", False), terrain=False)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--splits", nargs="*", default=None, help="subset of split names to generate")
    args = ap.parse_args()
    profile = PROFILES[args.profile]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(dict(profile=args.profile, spec=profile, suites=SUITES,
                       sensors={k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in v.items()}
                                for k, v in SENSORS.items()}), f, indent=1, default=float)
    jobs = build_jobs(args.out, profile, args.splits)
    print(f"[generate] {len(jobs)} scene jobs x {profile['frames']} frames with {args.workers} workers", flush=True)
    t0 = time.time()
    n_err = 0
    with Pool(args.workers) as pool:
        for i, (seed, err) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            if err:
                n_err += 1
                print(f"[generate] ERROR seed {seed}:\n{err}", flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"[generate] {i + 1}/{len(jobs)} scenes  {el / 60:.1f} min  eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.1f} min", flush=True)
    print(f"[generate] done, errors={n_err}, {(time.time() - t0) / 60:.1f} min", flush=True)
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
