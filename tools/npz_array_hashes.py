#!/usr/bin/env python3
"""Content hashes for the arrays inside .npz and .npy files, and a check against them.

Why: two .npz files holding identical arrays can have different file checksums
(older NumPy versions record the write time inside the container, and the
compression setting changes the bytes). This tool hashes the arrays themselves
(dtype, shape and raw bytes), so regenerated data can be verified bit for bit
against the originals.

Record the hashes of the original arrays:
    python tools/npz_array_hashes.py hash --root saved_latents --out latent_array_hashes.csv

Check regenerated arrays against them (same relative file names):
    python tools/npz_array_hashes.py check --root regenerated_latents --hashes latent_array_hashes.csv
    python tools/npz_array_hashes.py check --root sample_dir --hashes latent_array_hashes.csv --allow-missing

Requires NumPy. Object (pickled) arrays are refused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

EXTENSIONS = (".npz", ".npy")


def array_digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    if a.dtype.hasobject:
        raise ValueError("object arrays are not supported")
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(b"|")
    h.update(",".join(str(n) for n in a.shape).encode("ascii"))
    h.update(b"|")
    h.update(a.reshape(-1).view(np.uint8))
    return h.hexdigest()


def iter_files(root: str) -> Iterator[Tuple[str, str]]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.lower().endswith(EXTENSIONS):
                full = os.path.join(dirpath, fn)
                yield os.path.relpath(full, root).replace(os.sep, "/"), full


def iter_arrays(full: str) -> Iterator[Tuple[str, np.ndarray]]:
    if full.lower().endswith(".npy"):
        yield "", np.load(full, allow_pickle=False)
        return
    with np.load(full, allow_pickle=False) as z:
        for name in sorted(z.files):
            yield name, z[name]


def describe(arr: np.ndarray) -> Tuple[str, str]:
    return arr.dtype.str, "x".join(str(n) for n in arr.shape) or "scalar"


def cmd_hash(args: argparse.Namespace) -> int:
    rows: List[Tuple[str, str, str, str, str]] = []
    for rel, full in iter_files(args.root):
        for name, arr in iter_arrays(full):
            dtype, shape = describe(arr)
            rows.append((rel, name, dtype, shape, array_digest(arr)))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "array", "dtype", "shape", "sha256"])
        w.writerows(rows)
    files = len({r[0] for r in rows})
    print(f"hashed {len(rows):,} arrays in {files:,} files -> {args.out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    expected: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    with open(args.hashes, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            expected[(row["file"], row["array"])] = (row["dtype"], row["shape"], row["sha256"])
    seen = set()
    match = mismatch = extra = 0
    mismatches: List[str] = []
    for rel, full in iter_files(args.root):
        for name, arr in iter_arrays(full):
            key = (rel, name)
            dtype, shape = describe(arr)
            if key not in expected:
                extra += 1
                mismatches.append(f"EXTRA     {rel}::{name}")
                continue
            seen.add(key)
            if (dtype, shape, array_digest(arr)) == expected[key]:
                match += 1
            else:
                mismatch += 1
                e = expected[key]
                mismatches.append(f"DIFFERENT {rel}::{name}  got {dtype} {shape}, expected {e[0]} {e[1]}")
    missing_keys = sorted(set(expected) - seen)
    if args.allow_missing:
        files_present = {k[0] for k in seen}
        missing_keys = [k for k in missing_keys if k[0] in files_present]
    for line in mismatches[:50]:
        print(line)
    if len(mismatches) > 50:
        print(f"... {len(mismatches) - 50} more")
    for f, a in missing_keys[:20]:
        print(f"MISSING   {f}::{a}")
    if len(missing_keys) > 20:
        print(f"... {len(missing_keys) - 20} more missing")
    print(f"\n{match:,} identical, {mismatch:,} different, {extra:,} not in the hash list, "
          f"{len(missing_keys):,} missing")
    return 0 if mismatch == 0 and extra == 0 and not missing_keys else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Hash arrays inside .npz/.npy files, or check them against a hash list.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash", help="record content hashes")
    h.add_argument("--root", required=True)
    h.add_argument("--out", required=True)
    c = sub.add_parser("check", help="compare arrays against recorded hashes")
    c.add_argument("--root", required=True)
    c.add_argument("--hashes", required=True)
    c.add_argument("--allow-missing", action="store_true",
                   help="only check files that are present (for a regenerated sample); arrays missing "
                        "from a present file still fail")
    args = ap.parse_args(argv)
    return cmd_hash(args) if args.cmd == "hash" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
