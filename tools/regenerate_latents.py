#!/usr/bin/env python3
"""Replay frozen latent functions only; no scene rendering, scoring or inference.

The accepted-slot inventory and original namespace/seed code define the replay.
Scientific generator modules are imported unchanged from workspace/.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
MODULE = None
DEST = None


def initialize(study, out):
    global MODULE, DEST
    stage, filename = {
        "first_prospective": ("strengthening_2026_10_18", "generate_study3.py"),
        "replication": ("final_evidence_2026_10_20", "generate_study4.py"),
    }[study]
    folder = REPO / "workspace" / stage / "code"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("released_latent_generator", folder / filename)
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)
    MODULE.MODE = "holdout"
    MODULE.gen.rng = MODULE.phase_rng
    DEST = Path(out)
    for name in ["common", "common_afast", "faults", "kinafford.sim.scene"]:
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(REPO), name


def replay(task):
    import numpy as np
    scene, slot, rough = task
    g = MODULE.gen
    cameras = [g.camera_latents(scene, slot, i) for i in range(3)]
    lidar = g.lidar_latents(scene, slot)
    arrays = {f"camera{i}_{k}": v for i, fields in enumerate(cameras) for k, v in fields.items()}
    arrays.update({"lidar_" + k: v for k, v in lidar.items()})
    g.npzwrite(DEST / f"{scene}_{slot}.npz", **arrays)
    count = 1
    specs = MODULE.specs(scene, slot, rough)
    if "novel" in specs:
        for cam, name in enumerate(MODULE.CAMERAS):
            entry = specs["novel"]["cam"][name]
            if "novel" in entry:
                fields = MODULE.novel_latents((160, 256), entry["novel"], MODULE.phase_rng(scene, slot, 60, cam))
                np.savez_compressed(DEST / f"novel_{scene}_{slot}_{cam}.npz", **fields)
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", choices=["first_prospective", "replication"], required=True)
    ap.add_argument("--slots", type=Path, required=True)
    ap.add_argument("--identities", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    assert not args.out.exists(), "Replay must use a new output directory"
    assert 1 <= args.workers <= 4
    manifest = json.loads(args.identities.read_text())
    nominal = set(manifest["nominal_scene_ids"])
    rough_ids = set(manifest["rough_scene_ids"])
    rows = list(csv.DictReader(args.slots.open()))
    tasks = []
    for row in rows:
        if row["accepted"].lower() != "true":
            continue
        scene, slot = int(row["scene_id"]), int(row["frame_id"])
        rough = row["rough"].lower() == "true"
        assert scene in (rough_ids if rough else nominal), (scene, rough)
        tasks.append((scene, slot, rough))
    assert len(tasks) == len(set(tasks))
    args.out.mkdir(parents=True)
    start = time.time()
    files = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(args.study, str(args.out))) as pool:
        for n, count in enumerate(pool.map(replay, tasks), 1):
            files += count
            if n % 100 == 0:
                print(f"{args.study}: {n}/{len(tasks)} accepted bundles; {files} files", flush=True)
    receipt = {
        "study": args.study, "accepted_bundles": len(tasks), "files": files,
        "status": "REGENERATED_PENDING_KIT_ARRAY_CHECK", "wall_seconds": time.time() - start,
        "slots_sha256": hashlib.sha256(args.slots.read_bytes()).hexdigest(),
        "identities_sha256": hashlib.sha256(args.identities.read_bytes()).hexdigest(),
        "raw_saved_latents_read": False, "observations_rendered": 0, "neural_forwards": 0,
    }
    args.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
