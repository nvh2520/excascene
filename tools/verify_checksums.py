#!/usr/bin/env python3
"""Verify ExcaScene downloads against their checksums (Windows, macOS and Linux).

Check the downloaded deposit files against SHA256SUMS.txt:
    python verify_checksums.py --dir path/to/downloads
    python verify_checksums.py --dir path/to/downloads --only excascene_checkpoints_v1.0.0.zip

Check extracted files against MANIFEST.csv (every member, with size and SHA-256):
    python verify_checksums.py --manifest path/to/MANIFEST.csv --extracted path/to/data
    python verify_checksums.py --manifest MANIFEST.csv --extracted data --archive excascene_checkpoints_v1.0.0.zip

Missing files are reported but only count as failures when they were requested
with --only, or when --strict is given (useful after downloading everything).
Only the Python standard library is used (Python 3.8 or newer).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from typing import List, Optional, Sequence

LINE_RE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")


def sha256_file(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_sums(folder: str, sums_name: str, only: List[str], strict: bool) -> int:
    sums_path = os.path.join(folder, sums_name)
    if not os.path.isfile(sums_path):
        print(f"not found: {sums_path}")
        return 2
    ok = bad = missing = 0
    wanted = set(only)
    with open(sums_path, encoding="utf-8") as fh:
        entries = []
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            m = LINE_RE.match(line)
            if not m:
                print(f"unreadable line in {sums_name}: {line[:80]}")
                return 2
            entries.append((m.group(2), m.group(1).lower()))
    for name, digest in entries:
        if wanted and name not in wanted:
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            print(f"MISSING  {name}")
            missing += 1
            continue
        if sha256_file(path) == digest:
            print(f"OK       {name}")
            ok += 1
        else:
            print(f"FAILED   {name}")
            bad += 1
    unknown = wanted - {n for n, _ in entries}
    for name in sorted(unknown):
        print(f"NOT LISTED  {name}")
    print(f"\n{ok} OK, {bad} failed, {missing} missing")
    missing_counts = strict or bool(wanted)
    return 1 if bad or unknown or (missing and missing_counts) else 0


def verify_manifest(manifest: str, extracted: str, archive: Optional[str], strict: bool) -> int:
    ok = bad = missing = 0
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if archive and row["archive"] != archive:
                continue
            path = os.path.join(extracted, *row["member"].split("/"))
            if not os.path.isfile(path):
                missing += 1
                if missing <= 20:
                    print(f"MISSING  {row['member']}")
                continue
            if os.path.getsize(path) != int(row["bytes"]) or sha256_file(path) != row["sha256"].lower():
                bad += 1
                print(f"FAILED   {row['member']}")
            else:
                ok += 1
    if missing > 20:
        print(f"... {missing - 20} more missing")
    print(f"\n{ok} OK, {bad} failed, {missing} missing")
    return 1 if bad or (missing and strict) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify ExcaScene files against SHA256SUMS.txt or MANIFEST.csv.")
    ap.add_argument("--dir", help="folder with the downloaded deposit files")
    ap.add_argument("--sums", default="SHA256SUMS.txt", help="checksum file name inside --dir")
    ap.add_argument("--only", nargs="*", default=[], help="check only these file names")
    ap.add_argument("--manifest", help="MANIFEST.csv from the deposit")
    ap.add_argument("--extracted", help="folder where the archives were extracted")
    ap.add_argument("--archive", help="with --manifest: check only members of this archive")
    ap.add_argument("--strict", action="store_true", help="treat missing files as failures")
    args = ap.parse_args(argv)
    if args.manifest:
        if not args.extracted:
            ap.error("--manifest needs --extracted")
        return verify_manifest(args.manifest, args.extracted, args.archive, args.strict)
    if not args.dir:
        ap.error("give --dir, or --manifest with --extracted")
    return verify_sums(args.dir, args.sums, args.only, args.strict)


if __name__ == "__main__":
    sys.exit(main())
