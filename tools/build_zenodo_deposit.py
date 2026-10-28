#!/usr/bin/env python3
"""Build the ExcaScene Zenodo deposit from a release candidate, reproducibly.

Three steps:
  plan      resolve the manifest; check content rules, overlaps and Zenodo limits;
            list files of the release candidate that no archive takes
  pack      write the zip archives and MANIFEST.csv (every member with size and SHA-256)
  finalize  add README.md and LICENSE.txt, write SHA256SUMS.txt, re-check limits

  python tools/build_zenodo_deposit.py plan     --root RC --manifest deposit_manifest.csv --out DEPOSIT
  python tools/build_zenodo_deposit.py pack     --root RC --manifest deposit_manifest.csv --out DEPOSIT
  python tools/build_zenodo_deposit.py finalize --out DEPOSIT --readme README.md --license LICENSE.txt

Reports (plan.json, UNASSIGNED.csv, ARCHIVE_TABLE.md, pack_receipt.json) go to
DEPOSIT_reports/ next to the deposit, never into the deposit itself.

Manifest: CSV with the header
  archive,prefix,source,include,exclude,compression,description
  archive      zip file name, e.g. excascene_checkpoints_v1.0.0.zip
  prefix       folder inside the archive, e.g. checkpoints/replication_42epoch
  source       file or folder relative to --root
  include      ';'-separated globs relative to source (default '*')
  exclude      ';'-separated globs relative to source (optional)
  compression  auto | store | deflate  (auto stores formats that are already compressed)
  description  one line, used for the archive table
Globs follow fnmatch rules and are tested against the path relative to source and
against the file name, so '*' also matches '/' and '*.pt' matches at any depth.
Rows whose archive cell starts with '#' are ignored.

Archives are deterministic: members are sorted, timestamps fixed at 1980-01-01,
permissions normalised. Content rules are shared with tools/release_audit.py;
REVIEW findings must be fixed or acknowledged in the allowlist
(--allowlist, same format as the audit, paths written as archive.zip::member).

Only the Python standard library is used (Python 3.8 or newer).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import posixpath
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True  # keep tools/ free of __pycache__
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_audit as ra  # noqa: E402

MiB = 1024 * 1024
GB = 1000 ** 3
CHUNK = 8 * MiB
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
REQUIRED_COLUMNS = ("archive", "prefix", "source", "include", "exclude", "compression", "description")
ARCHIVE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.zip$")
ALREADY_COMPRESSED = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".gz", ".tgz", ".bz2", ".xz",
                      ".zst", ".zip", ".7z", ".mp4", ".pdf"}
DEFAULT_EXCLUDES = ["*__pycache__*", "*.pyc", "*.DS_Store", "*.ipynb_checkpoints*", ".git/*", "*/.git/*"]
ROOT_FILES = ("README.md", "LICENSE.txt", "MANIFEST.csv", "SHA256SUMS.txt")


@dataclass
class Row:
    archive: str
    prefix: str
    source: str
    include: List[str]
    exclude: List[str]
    compression: str
    description: str
    line: int


@dataclass
class Item:
    archive: str
    member: str
    src_abs: str
    src_rel: str
    size: int
    compression: str


@dataclass
class Resolution:
    items: List[Item] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    default_excluded: int = 0


# ---------------------------------------------------------------- manifest

def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def split_globs(cell: str) -> List[str]:
    return [g.strip() for g in cell.split(";") if g.strip()]


def bad_relative(path: str) -> bool:
    norm = path.replace("\\", "/")
    return norm.startswith("/") or bool(re.match(r"^[A-Za-z]:", norm)) or ".." in norm.split("/")


def read_manifest(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            die(f"{path}: missing columns {missing}")
        problems: List[str] = []
        for n, rec in enumerate(reader, start=2):
            vals = {k: (rec.get(k) or "").strip() for k in REQUIRED_COLUMNS}
            if not any(vals.values()) or vals["archive"].startswith("#"):
                continue
            where = f"{path}:{n}"
            if not ARCHIVE_NAME_RE.match(vals["archive"]):
                problems.append(f"{where}: archive must be a plain .zip file name")
            src = vals["source"]
            if not src or src.upper().startswith("TODO") or ra.PLACEHOLDER_RE.search(src):
                problems.append(f"{where}: source is not filled in")
            elif bad_relative(src):
                problems.append(f"{where}: source must be relative to --root, without '..'")
            prefix = vals["prefix"].strip("/")
            if bad_relative(prefix) or ra.PLACEHOLDER_RE.search(prefix):
                problems.append(f"{where}: prefix must be a relative folder without '..'")
            comp = (vals["compression"] or "auto").lower()
            if comp not in ("auto", "store", "deflate"):
                problems.append(f"{where}: compression must be auto, store or deflate")
            rows.append(Row(vals["archive"], prefix, src, split_globs(vals["include"]) or ["*"],
                            split_globs(vals["exclude"]), comp, vals["description"], n))
        if problems:
            die("manifest problems:\n  " + "\n  ".join(problems))
    if not rows:
        die(f"{path}: no manifest rows")
    return rows


def matches(rel: str, globs: Sequence[str]) -> bool:
    base = posixpath.basename(rel)
    return any(fnmatch.fnmatchcase(rel, g) or fnmatch.fnmatchcase(base, g) for g in globs)


def walk(folder: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full):
                out.append((os.path.relpath(full, folder).replace(os.sep, "/"), full))
    return out


def resolve(rows: List[Row], root: str) -> Resolution:
    res = Resolution()
    root_real = os.path.realpath(root)
    for row in rows:
        src = os.path.realpath(os.path.join(root_real, row.source))
        try:
            inside = os.path.commonpath([root_real, src]) == root_real
        except ValueError:
            inside = False
        if not inside:
            res.errors.append(f"manifest line {row.line}: source resolves outside --root: {row.source}")
            continue
        if not os.path.exists(src):
            res.errors.append(f"manifest line {row.line}: source not found: {row.source}")
            continue
        if os.path.isfile(src):
            candidates = [(os.path.basename(src), src)]
            use_include = False
        else:
            candidates = walk(src)
            use_include = True
        taken = 0
        for rel, full in candidates:
            if matches(rel, DEFAULT_EXCLUDES):
                res.default_excluded += 1
                continue
            if use_include and not matches(rel, row.include):
                continue
            if matches(rel, row.exclude):
                continue
            member = posixpath.join(row.prefix, rel) if row.prefix else rel
            src_rel = os.path.relpath(full, root_real).replace(os.sep, "/")
            res.items.append(Item(row.archive, member, full, src_rel, os.path.getsize(full), row.compression))
            taken += 1
        if taken == 0:
            res.errors.append(f"manifest line {row.line}: no files selected from {row.source}")
    seen_src: Dict[str, Item] = {}
    seen_member: Dict[Tuple[str, str], Item] = {}
    for it in res.items:
        real = os.path.realpath(it.src_abs)
        if real in seen_src:
            prev = seen_src[real]
            res.errors.append(f"file packed twice: {it.src_rel} -> {prev.archive}::{prev.member} and {it.archive}::{it.member}")
        seen_src[real] = it
        key = (it.archive, it.member)
        if key in seen_member:
            res.errors.append(f"duplicate member {it.archive}::{it.member}")
        seen_member[key] = it
    return res


def descriptions(rows: List[Row]) -> Dict[str, str]:
    out: Dict[str, List[str]] = {}
    for r in rows:
        if r.description and r.description not in out.setdefault(r.archive, []):
            out[r.archive].append(r.description)
    return {k: "; ".join(v) for k, v in out.items()}


def check_items(items: List[Item], allowlist, allowed_emails) -> List[ra.Finding]:
    findings: List[ra.Finding] = []
    for it in items:
        key = f"{it.archive}::{it.member}"
        # Check both the packed name and the original location: a prefix can hide
        # a telling folder name (for example goose_ex/ packed as real_image_check/).
        seen_rules = set()
        for path in (key, it.src_rel):
            for sev, rule, detail in ra.path_findings(path, "zenodo"):
                if rule in seen_rules:
                    continue
                seen_rules.add(rule)
                where = "" if path == key else f" (source: {it.src_rel})"
                findings.append(ra.Finding(sev, rule, key, detail + where))
        if it.size <= 5 * MiB:
            with open(it.src_abs, "rb") as fh:
                data = fh.read()
            if ra.looks_text(it.member, data[:8192]):
                ra.scan_text(data.decode("utf-8", "replace"), key, findings,
                             allowed_emails=allowed_emails, local_path_severity="REVIEW")
    ra.apply_allowlist(findings, allowlist)
    return findings


def report_dir_for(out: str) -> str:
    d = os.path.abspath(out).rstrip("/\\") + "_reports"
    os.makedirs(d, exist_ok=True)
    return d


def summarise(items: List[Item]) -> Dict[str, Dict[str, int]]:
    s: Dict[str, Dict[str, int]] = {}
    for it in items:
        a = s.setdefault(it.archive, {"files": 0, "bytes": 0})
        a["files"] += 1
        a["bytes"] += it.size
    return s


def print_findings(findings: List[ra.Finding]) -> List[ra.Finding]:
    block = ra.blocking(findings)
    ack = [f for f in findings if f.allowed_reason]
    for f in sorted(block, key=lambda x: (x.severity, x.rule, x.path)):
        loc = f"{f.path}:{f.line}" if f.line else f.path
        print(f"  [{f.severity}] {f.rule}  {loc}  {f.detail}")
    if ack:
        print(f"  ({len(ack)} finding(s) acknowledged in the allowlist)")
    return block


def prepare(args) -> Tuple[List[Row], Resolution, List[ra.Finding]]:
    rows = read_manifest(args.manifest)
    res = resolve(rows, args.root)
    allow = ra.load_allowlist(args.allowlist) if args.allowlist else []
    emails = {e.strip().lower() for e in args.allowed_emails.split(",") if e.strip()}
    findings = check_items(res.items, allow, emails)
    return rows, res, findings


# ---------------------------------------------------------------- plan

def cmd_plan(args) -> int:
    rows, res, findings = prepare(args)
    summary = summarise(res.items)
    total = sum(a["bytes"] for a in summary.values())
    root_files = len(summary) + len(ROOT_FILES)
    print(f"Plan for {args.out}")
    for name in sorted(summary):
        a = summary[name]
        print(f"  {name:<48} {a['files']:>9,} files  {ra.human_bytes(a['bytes']):>10} (before compression)")
    print(f"  total {ra.human_bytes(total)} in {len(res.items):,} files; deposit root will hold {root_files} files")
    status = 0
    if res.errors:
        status = 1
        print("Errors:")
        for e in res.errors:
            print(f"  {e}")
    if root_files > args.max_files:
        status = 1
        print(f"Too many deposit files: {root_files} > {args.max_files}")
    if total > args.quota_gb * GB:
        print(f"Note: the uncompressed total exceeds {args.quota_gb:g} GB; check the packed size, and request "
              "the one-time Zenodo quota increase if it still exceeds the limit")
    print("Content findings:")
    block = print_findings(findings)
    if not block:
        print("  none blocking")
    else:
        status = 1
    # files of the release candidate that no archive takes
    taken = {os.path.realpath(i.src_abs) for i in res.items}
    unassigned: List[Tuple[str, int]] = []
    for rel, full in walk(os.path.realpath(args.root)):
        if matches(rel, DEFAULT_EXCLUDES) or os.path.realpath(full) in taken:
            continue
        unassigned.append((rel, os.path.getsize(full)))
    rdir = report_dir_for(args.out)
    with open(os.path.join(rdir, "UNASSIGNED.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes"])
        for rel, size in sorted(unassigned, key=lambda x: (-x[1], x[0])):
            w.writerow([rel, size])
    with open(os.path.join(rdir, "plan.json"), "w", encoding="utf-8") as fh:
        json.dump({"archives": summary, "total_bytes": total, "files": len(res.items),
                   "root_files": root_files, "errors": res.errors,
                   "unassigned_files": len(unassigned),
                   "unassigned_bytes": sum(s for _, s in unassigned),
                   "default_excluded": res.default_excluded,
                   "blocking_findings": [f.__dict__ for f in block]}, fh, indent=2)
    print(f"Unassigned files: {len(unassigned):,} ({ra.human_bytes(sum(s for _, s in unassigned))}); "
          f"see {os.path.join(rdir, 'UNASSIGNED.csv')}")
    print("Plan OK" if status == 0 else "Plan has problems; fix them before pack")
    return status


# ---------------------------------------------------------------- pack

def set_level(zinfo: zipfile.ZipInfo, level: int) -> None:
    for attr in ("compress_level", "_compresslevel"):
        try:
            setattr(zinfo, attr, level)
        except AttributeError:
            pass


def cmd_pack(args) -> int:
    rows, res, findings = prepare(args)
    if res.errors:
        die("manifest errors (run plan for details):\n  " + "\n  ".join(res.errors))
    block = ra.blocking(findings)
    if block:
        print("Blocking content findings:")
        print_findings(findings)
        die("fix or acknowledge these before packing")
    os.makedirs(args.out, exist_ok=True)
    groups: Dict[str, List[Item]] = {}
    for it in res.items:
        groups.setdefault(it.archive, []).append(it)
    for name in groups:
        target = os.path.join(args.out, name)
        if os.path.exists(target) and not args.overwrite:
            die(f"{target} exists; use --overwrite to rebuild")
    manifest_rows: List[Tuple[str, str, int, str, str]] = []
    archives: Dict[str, Dict[str, object]] = {}
    started = time.time()
    for name in sorted(groups):
        target = os.path.join(args.out, name)
        tmp = target + ".partial"
        items = sorted(groups[name], key=lambda i: i.member)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for it in items:
                ext = os.path.splitext(it.member.lower())[1]
                store = it.compression == "store" or (it.compression == "auto" and ext in ALREADY_COMPRESSED)
                zinfo = zipfile.ZipInfo(it.member, date_time=FIXED_DATE)
                zinfo.create_system = 3
                zinfo.external_attr = (0o100644 & 0xFFFF) << 16
                zinfo.compress_type = zipfile.ZIP_STORED if store else zipfile.ZIP_DEFLATED
                if not store:
                    set_level(zinfo, args.level)
                zinfo.file_size = it.size
                digest = hashlib.sha256()
                with open(it.src_abs, "rb") as src, zf.open(zinfo, "w") as dst:
                    while True:
                        block_ = src.read(CHUNK)
                        if not block_:
                            break
                        digest.update(block_)
                        dst.write(block_)
                manifest_rows.append((name, it.member, it.size, digest.hexdigest(), it.src_rel))
        with zipfile.ZipFile(tmp) as check:
            count = sum(1 for i in check.infolist() if not i.is_dir())
            if count != len(items):
                die(f"{name}: wrote {count} members, expected {len(items)}")
            if args.test:
                bad = check.testzip()
                if bad:
                    die(f"{name}: CRC check failed for {bad}")
        os.replace(tmp, target)
        size = os.path.getsize(target)
        archives[name] = {"members": len(items), "bytes": size,
                          "uncompressed_bytes": sum(i.size for i in items),
                          "sha256": ra.sha256_file(target)}
        print(f"  wrote {name}: {len(items):,} files, {ra.human_bytes(size)}")
    with open(os.path.join(args.out, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["archive", "member", "bytes", "sha256", "source"])
        for row in sorted(manifest_rows):
            w.writerow(row)
    rdir = report_dir_for(args.out)
    desc = descriptions(rows)
    with open(os.path.join(rdir, "ARCHIVE_TABLE.md"), "w", encoding="utf-8") as fh:
        fh.write("| File | Size | Files inside | Contents |\n|---|---:|---:|---|\n")
        for name in sorted(archives):
            a = archives[name]
            fh.write(f"| `{name}` | {ra.human_bytes(a['bytes'])} | {a['members']:,} | {desc.get(name, '')} |\n")
    with open(os.path.join(rdir, "pack_receipt.json"), "w", encoding="utf-8") as fh:
        json.dump({"tool": "build_zenodo_deposit.py pack", "python": sys.version.split()[0],
                   "zlib": getattr(zipfile.zlib, "ZLIB_RUNTIME_VERSION", "unknown") if zipfile.zlib else "none",
                   "compress_level": args.level, "manifest": os.path.abspath(args.manifest),
                   "archives": archives, "members": len(manifest_rows),
                   "seconds": round(time.time() - started, 1),
                   "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=2)
    total = sum(a["bytes"] for a in archives.values())
    print(f"Packed {len(archives)} archives, {ra.human_bytes(total)}. Paste {os.path.join(rdir, 'ARCHIVE_TABLE.md')} "
          "into the deposit README, then run finalize.")
    return 0


# ---------------------------------------------------------------- finalize

def cmd_finalize(args) -> int:
    out = args.out
    if not os.path.isfile(os.path.join(out, "MANIFEST.csv")):
        die("MANIFEST.csv not found; run pack first")
    for src, name in ((args.readme, "README.md"), (args.license, "LICENSE.txt")):
        with open(src, encoding="utf-8") as fh:
            text = fh.read()
        left = sorted(set(ra.PLACEHOLDER_RE.findall(text)))
        deferred = set(getattr(args, "defer_placeholders", "").split(","))
        unresolved = [x for x in left if x not in deferred]
        if unresolved:
            die(f"{src} still has placeholders: {', '.join(unresolved)}")
        if left:
            print("AUTHOR FIELDS DEFERRED; NOT PUBLICATION-READY: " + ", ".join(left))
        shutil.copyfile(src, os.path.join(out, name))
    names = sorted(os.listdir(out))
    subdirs = [n for n in names if os.path.isdir(os.path.join(out, n))]
    if subdirs:
        die(f"deposit folder must be flat; remove folders: {subdirs}")
    stray = [n for n in names if n != "SHA256SUMS.txt" and not n.endswith(".zip") and n not in ROOT_FILES]
    if stray:
        die(f"unexpected files in the deposit folder: {stray}")
    files = [n for n in names if n != "SHA256SUMS.txt"]
    with open(os.path.join(out, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        for n in files:
            fh.write(f"{ra.sha256_file(os.path.join(out, n))}  {n}\n")
    files.append("SHA256SUMS.txt")
    total = sum(os.path.getsize(os.path.join(out, n)) for n in files)
    print(f"Deposit: {len(files)} files, {ra.human_bytes(total)}")
    status = 0
    if len(files) > args.max_files:
        print(f"Too many files for one Zenodo record: {len(files)} > {args.max_files}")
        status = 1
    if total > args.quota_gb * GB:
        print(f"Exceeds {args.quota_gb:g} GB: request Zenodo's one-time quota increase before uploading "
              "(do not split one dataset into several records)")
        status = 1
    print("Next: python tools/release_audit.py --mode zenodo --root " + out + " --test-zips")
    return status


# ---------------------------------------------------------------- main

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the ExcaScene Zenodo deposit.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "pack"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True, help="release candidate folder")
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True, help="deposit folder (flat; uploaded to Zenodo as is)")
        p.add_argument("--allowlist", help="acknowledged findings, same format as release_audit")
        p.add_argument("--allowed-emails", default=",".join(sorted(ra.DEFAULT_ALLOWED_EMAILS)))
        p.add_argument("--max-files", type=int, default=100)
        p.add_argument("--quota-gb", type=float, default=50.0)
        if name == "pack":
            p.add_argument("--overwrite", action="store_true")
            p.add_argument("--test", action="store_true", help="read back every archive and check CRCs")
            p.add_argument("--level", type=int, default=6, help="deflate level 1-9 (default 6)")
    f = sub.add_parser("finalize")
    f.add_argument("--out", required=True)
    f.add_argument("--readme", required=True)
    f.add_argument("--license", required=True)
    f.add_argument("--defer-placeholders", default="", help="Explicit author-field deferral for private review only; omit for publication")
    f.add_argument("--max-files", type=int, default=100)
    f.add_argument("--quota-gb", type=float, default=50.0)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "pack":
        return cmd_pack(args)
    return cmd_finalize(args)


if __name__ == "__main__":
    sys.exit(main())
