#!/usr/bin/env python3
"""Pre-publication audit for the ExcaScene code repository and Zenodo deposit.

It looks for things that must not be published, or that would make the release
unusable: unfilled double-brace placeholders, local absolute paths, credentials,
unexpected e-mail addresses, excluded content (GOOSE-Ex images, annotation and
governance material, AI-tool state), files that do not belong in a git
repository, and Zenodo limits and checksums.

Usage
    python tools/release_audit.py --mode github --root .
    python tools/release_audit.py --mode zenodo --root path/to/deposit

Severity
    FAIL    must be fixed before publishing.
    REVIEW  fix it, or acknowledge it in the allowlist with a reason.
    INFO    shown for information (for example deferred placeholders).
The exit status is 0 only when there is no FAIL and no unacknowledged REVIEW.

Allowlist (default: <root>/.release-audit-allow.txt), one entry per line:
    <path glob> | <rule id or *> | <reason>
Paths are relative to --root, with forward slashes. A member of a zip archive
is written as  archive.zip::path/inside/archive . Globs follow fnmatch rules,
so '*' also matches '/'. A REVIEW finding can be acknowledged with '*' or its
rule id; a FAIL finding only with its exact rule id. Lines starting with '#'
are comments.

Only the Python standard library is used (Python 3.8 or newer).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

MiB = 1024 * 1024
GB = 1000 ** 3

TEXT_EXTENSIONS = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".cfg", ".ini", ".csv", ".tsv", ".cff", ".sh", ".bash", ".bat", ".ps1", ".tex",
    ".bib", ".html", ".htm", ".js", ".ts", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h",
    ".hpp", ".cmake", ".diff", ".patch", ".xml", ".svg", ".log", ".ipynb",
}
TEXT_BASENAMES = {
    "license", "licence", "notice", "copying", "authors", "readme", "changelog",
    "dockerfile", "makefile", ".gitignore", ".gitattributes", ".release-audit-allow",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".tflite", ".pb"}
BINARY_EXTENSIONS = {".so", ".dll", ".dylib", ".pyd", ".o", ".a", ".obj", ".lib", ".exe", ".whl", ".egg"}
DATA_EXTENSIONS = {
    ".npz", ".npy", ".pkl", ".pickle", ".parquet", ".feather", ".h5", ".hdf5", ".mat",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".zst",
}
KEY_FILE_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".keystore", ".jks"}
CACHE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TOOL_STATE_DIRS = {".claude", ".codex", ".cursor", ".aider", ".continue", ".vscode-server"}
SKIP_DIRS = {".git"}
SELF_FILES = {"tools/release_audit.py", "tools/build_zenodo_deposit.py"}

DEFAULT_ALLOWED_EMAILS = {"jinhong@inje.ac.kr", "nviethung1998@live.inje.ac.kr", "git@github.com"}
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "users.noreply.github.com", "noreply.github.com"}

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.~-])("
    r"/media/[^\s'\"<>|,;)]+"
    r"|/home/[A-Za-z0-9_.-]+"
    r"|/root/"
    r"|/mnt/[A-Za-z0-9_.-]+"
    r"|/Users/[A-Za-z0-9_.-]+/"
    r"|[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"
    r")"
)
EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})"
)
SECRET_RULES: List[Tuple[str, "re.Pattern[str]", str]] = [
    ("SECRET_ANTHROPIC_KEY", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "FAIL"),
    ("SECRET_OPENAI_KEY", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}"), "FAIL"),
    ("SECRET_GITHUB_TOKEN", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})"), "FAIL"),
    ("SECRET_AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "FAIL"),
    ("SECRET_GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"), "FAIL"),
    ("SECRET_HF_TOKEN", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "FAIL"),
    ("SECRET_SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "FAIL"),
    ("SECRET_PRIVATE_KEY", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"), "FAIL"),
    ("SECRET_ASSIGNMENT", re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret"
        r"|password|passwd)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"), "REVIEW"),
]
SENSITIVE_PATH_RE = re.compile(
    r"(?:^|[/_.\-:])q2(?:[/_.\-]|$)"
    r"|vlm[_\-]?(?:label|annot)"
    r"|blind[_\-]?subset"
    r"|realism[_\-]?check"
    r"|author[_\-]?(?:response|decision)"
    r"|decision[_\-]?delta"
    r"|response[_\-]?form"
    r"|questionnaire"
    r"|integrity[_\-]?(?:dossier|review|record|decision)"
    r"|dossier"
    r"|transcript"
    r"|session[_\-]?log",
    re.IGNORECASE,
)
REAL_PATH_RE = re.compile(r"(?:^|[/_\-:])real(?:[/_\-.]|$)", re.IGNORECASE)
SHA_LINE_RE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")


@dataclass
class Finding:
    severity: str
    rule: str
    path: str
    detail: str
    line: Optional[int] = None
    allowed_reason: Optional[str] = None


# ---------------------------------------------------------------- helpers

def human_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} TB"


def sha256_file(path: str, chunk: int = 8 * MiB) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def looks_text(path: str, head: bytes) -> bool:
    base = os.path.basename(path.split("::")[-1]).lower()
    stem, ext = os.path.splitext(base)
    if b"\x00" in head:
        return False
    if ext in TEXT_EXTENSIONS or base in TEXT_BASENAMES or stem in TEXT_BASENAMES:
        return True
    if ext == "" and head:
        try:
            head.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    return False


def load_allowlist(path: Optional[str]) -> List[Tuple[str, str, str]]:
    entries: List[Tuple[str, str, str]] = []
    if not path or not os.path.isfile(path):
        return entries
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3 or not all(parts):
                raise SystemExit(f"{path}:{n}: expected '<path glob> | <rule id or *> | <reason>'")
            entries.append((parts[0], parts[1], parts[2]))
    return entries


def apply_allowlist(findings: List[Finding], allowlist: Sequence[Tuple[str, str, str]]) -> None:
    for f in findings:
        if f.severity not in ("FAIL", "REVIEW"):
            continue
        for glob, rule, reason in allowlist:
            if not fnmatch.fnmatchcase(f.path, glob):
                continue
            if rule == f.rule or (rule == "*" and f.severity == "REVIEW"):
                f.allowed_reason = reason
                break


def blocking(findings: Iterable[Finding]) -> List[Finding]:
    return [f for f in findings if f.severity in ("FAIL", "REVIEW") and not f.allowed_reason]


# ---------------------------------------------------------------- rules

def path_findings(path: str, mode: str) -> List[Tuple[str, str, str]]:
    """Rules that depend only on the file name and location."""
    out: List[Tuple[str, str, str]] = []
    inner = path.split("::")[-1]
    parts = inner.split("/")
    name = parts[-1]
    low = name.lower()
    ext = os.path.splitext(low)[1]
    dirs = [p.lower() for p in parts[:-1]]
    if any(d in TOOL_STATE_DIRS for d in dirs):
        out.append(("FAIL", "EXCLUDED_TOOL_STATE", "AI-tool or editor state directory"))
    if (low == ".env" or low.startswith(".env.") or ext in KEY_FILE_EXTENSIONS
            or low.startswith(("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"))):
        out.append(("FAIL", "EXCLUDED_CREDENTIAL_FILE", "credential or key file"))
    if ext in IMAGE_EXTENSIONS:
        if "goose" in path.lower():
            out.append(("FAIL", "EXCLUDED_GOOSE_IMAGE", "GOOSE-Ex images are not redistributed"))
        elif REAL_PATH_RE.search(path):
            out.append(("REVIEW", "REAL_IMAGE", "image under a 'real' path; confirm it is not GOOSE-Ex"))
    if SENSITIVE_PATH_RE.search(path):
        out.append(("REVIEW", "SENSITIVE_PATH", "annotation, governance or transcript material"))
    if mode == "github":
        if any(d in CACHE_DIRS for d in dirs) or ext == ".pyc" or low == ".ds_store":
            out.append(("FAIL", "CACHE_FILE", "cache files do not belong in git"))
        if ext in WEIGHT_EXTENSIONS:
            out.append(("FAIL", "WEIGHTS_IN_GIT", "checkpoints belong in the Zenodo deposit"))
        if ext in BINARY_EXTENSIONS:
            out.append(("FAIL", "BINARY_IN_GIT", "compiled binaries are built from source, not committed"))
        if ext in DATA_EXTENSIONS:
            out.append(("REVIEW", "DATA_IN_GIT", "data arrays or archives normally belong in the Zenodo deposit"))
    return out


def scan_text(text: str, path: str, findings: List[Finding], *, allowed_emails: Set[str],
              check_emails: bool = True, local_path_severity: str = "FAIL",
              placeholder_severity: str = "FAIL", deferred: Set[str] = frozenset(),
              max_per_rule: int = 5) -> None:
    counts: Dict[str, int] = {}

    def add(sev: str, rule: str, line_no: int, detail: str) -> None:
        c = counts.get(rule, 0)
        counts[rule] = c + 1
        if c < max_per_rule:
            findings.append(Finding(sev, rule, path, detail, line_no))

    for line_no, line in enumerate(text.splitlines(), 1):
        for m in PLACEHOLDER_RE.finditer(line):
            if m.group(1) in deferred:
                add("INFO", "PLACEHOLDER_DEFERRED", line_no, m.group(0))
            else:
                add(placeholder_severity, "PLACEHOLDER", line_no, m.group(0))
        for m in LOCAL_PATH_RE.finditer(line):
            add(local_path_severity, "LOCAL_PATH", line_no, m.group(1)[:120])
        for rule, rx, sev in SECRET_RULES:
            if rx.search(line):
                add(sev, rule, line_no, "possible credential (value not shown)")
        if check_emails:
            for m in EMAIL_RE.finditer(line):
                email = m.group(1).lower()
                domain = email.split("@", 1)[1]
                if email in allowed_emails or domain in ALLOWED_EMAIL_DOMAINS:
                    continue
                add("REVIEW", "EMAIL", line_no, email)
    for rule, c in counts.items():
        if c > max_per_rule:
            findings.append(Finding("INFO", rule + "_MORE", path,
                                    f"{c - max_per_rule} more occurrences of {rule} not listed"))


# ---------------------------------------------------------------- github mode

def git_listed_files(root: str) -> Optional[List[str]]:
    if not os.path.isdir(os.path.join(root, ".git")):
        return None
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted({p for p in out.decode("utf-8", "replace").split("\0") if p})


def walk_files(root: str) -> List[str]:
    rels: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rels.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return rels


def audit_github(root: str, args: argparse.Namespace, findings: List[Finding]) -> Dict[str, object]:
    rels = None if args.all_files else git_listed_files(root)
    source = "git (tracked and untracked, not ignored)" if rels is not None else "directory walk"
    if rels is None:
        rels = walk_files(root)
    total = 0
    for rel in rels:
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            continue
        size = os.path.getsize(full)
        total += size
        for sev, rule, detail in path_findings(rel, "github"):
            findings.append(Finding(sev, rule, rel, detail))
        if size > 50 * MiB:
            findings.append(Finding("FAIL", "LARGE_FILE", rel, f"{human_bytes(size)} (git warns above 50 MiB, GitHub blocks above 100 MiB)"))
        elif size > 10 * MiB:
            findings.append(Finding("REVIEW", "LARGE_FILE", rel, human_bytes(size)))
        if rel in SELF_FILES or size > 20 * MiB:
            continue
        with open(full, "rb") as fh:
            data = fh.read()
        if looks_text(rel, data[:8192]):
            scan_text(data.decode("utf-8", "replace"), rel, findings,
                      allowed_emails=args.allowed_email_set,
                      check_emails=not rel.startswith("third_party/"),
                      local_path_severity="FAIL", deferred=args.deferred)
    for req in ("README.md", "LICENSE", "NOTICE", "CITATION.cff", "THIRD_PARTY_NOTICES.md"):
        if not os.path.isfile(os.path.join(root, req)):
            findings.append(Finding("FAIL", "MISSING_FILE", req, "required file is missing"))
    zj = os.path.join(root, ".zenodo.json")
    if os.path.isfile(zj):
        try:
            with open(zj, encoding="utf-8") as fh:
                meta = json.load(fh)
            for key in ("title", "upload_type", "creators", "license", "description"):
                if key not in meta:
                    findings.append(Finding("FAIL", "ZENODO_JSON", ".zenodo.json", f"missing key '{key}'"))
            if meta.get("upload_type") != "software":
                findings.append(Finding("REVIEW", "ZENODO_JSON", ".zenodo.json", "upload_type is not 'software'"))
        except ValueError as exc:
            findings.append(Finding("FAIL", "ZENODO_JSON", ".zenodo.json", f"invalid JSON: {exc}"))
    else:
        findings.append(Finding("REVIEW", "MISSING_FILE", ".zenodo.json",
                                "without it Zenodo takes metadata from CITATION.cff or GitHub"))
    if total > 5 * GB:
        findings.append(Finding("FAIL", "REPO_SIZE", ".", f"{human_bytes(total)}; GitHub strongly recommends under 5 GB"))
    elif total > 1 * GB:
        findings.append(Finding("REVIEW", "REPO_SIZE", ".", f"{human_bytes(total)}; GitHub recommends under 1 GB"))
    return {"files_checked": len(rels), "bytes": total, "file_source": source}


# ---------------------------------------------------------------- zenodo mode

def parse_sha256sums(path: str, findings: List[Finding]) -> Dict[str, str]:
    sums: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            m = SHA_LINE_RE.match(line)
            if not m:
                findings.append(Finding("FAIL", "SHA256SUMS_FORMAT", "SHA256SUMS.txt", "unreadable line", n))
                continue
            sums[m.group(2)] = m.group(1).lower()
    return sums


def read_manifest_members(path: str) -> Dict[str, Set[str]]:
    import csv
    members: Dict[str, Set[str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            members.setdefault(row["archive"], set()).add(row["member"])
    return members


def audit_zenodo(root: str, args: argparse.Namespace, findings: List[Finding]) -> Dict[str, object]:
    files: List[str] = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            findings.append(Finding("FAIL", "ZENODO_SUBDIRECTORY", name,
                                    "a Zenodo record holds a flat list of files; pack folders into archives"))
        elif os.path.isfile(full):
            files.append(name)
    total = sum(os.path.getsize(os.path.join(root, f)) for f in files)
    if len(files) > args.max_files:
        findings.append(Finding("FAIL", "ZENODO_FILE_COUNT", ".", f"{len(files)} files; limit {args.max_files}"))
    quota = args.quota_gb * GB
    if total > quota:
        findings.append(Finding("FAIL", "ZENODO_TOTAL_SIZE", ".",
                                f"{human_bytes(total)} exceeds {args.quota_gb:g} GB; request the one-time "
                                "quota increase before uploading, do not split one dataset into several records"))
    for req in ("README.md", "LICENSE.txt", "SHA256SUMS.txt"):
        if req not in files:
            findings.append(Finding("FAIL", "MISSING_FILE", req, "required deposit file is missing"))
    if "MANIFEST.csv" not in files:
        findings.append(Finding("REVIEW", "MISSING_FILE", "MANIFEST.csv", "member-level inventory is missing"))
    leftovers = [f for f in files if f.endswith(".partial")]
    for f in leftovers:
        findings.append(Finding("FAIL", "PARTIAL_FILE", f, "incomplete archive left by an interrupted pack"))

    if "SHA256SUMS.txt" in files:
        sums = parse_sha256sums(os.path.join(root, "SHA256SUMS.txt"), findings)
        for name in files:
            if name != "SHA256SUMS.txt" and name not in sums:
                findings.append(Finding("FAIL", "UNLISTED_FILE", name, "not listed in SHA256SUMS.txt"))
        for name, digest in sorted(sums.items()):
            if name not in files:
                findings.append(Finding("FAIL", "LISTED_FILE_MISSING", name, "listed in SHA256SUMS.txt but absent"))
            elif not args.skip_hash and sha256_file(os.path.join(root, name)) != digest:
                findings.append(Finding("FAIL", "CHECKSUM_MISMATCH", name, "SHA-256 differs from SHA256SUMS.txt"))

    for name in files:
        full = os.path.join(root, name)
        for sev, rule, detail in path_findings(name, "zenodo"):
            findings.append(Finding(sev, rule, name, detail))
        if name.endswith(".zip") or os.path.getsize(full) > 20 * MiB:
            continue
        with open(full, "rb") as fh:
            data = fh.read()
        if looks_text(name, data[:8192]):
            scan_text(data.decode("utf-8", "replace"), name, findings,
                      allowed_emails=args.allowed_email_set, local_path_severity="REVIEW",
                      deferred=args.deferred)

    manifest = None
    if "MANIFEST.csv" in files:
        try:
            manifest = read_manifest_members(os.path.join(root, "MANIFEST.csv"))
        except (KeyError, ValueError) as exc:
            findings.append(Finding("FAIL", "MANIFEST_FORMAT", "MANIFEST.csv", f"unreadable: {exc}"))
    members_total = 0
    for name in files:
        if not name.endswith(".zip"):
            continue
        full = os.path.join(root, name)
        try:
            zf = zipfile.ZipFile(full)
        except zipfile.BadZipFile as exc:
            findings.append(Finding("FAIL", "BAD_ZIP", name, str(exc)))
            continue
        with zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            members_total += len(infos)
            names = set()
            for info in infos:
                member = info.filename
                names.add(member)
                key = f"{name}::{member}"
                if member.startswith(("/", "\\")) or ".." in member.replace("\\", "/").split("/"):
                    findings.append(Finding("FAIL", "ZIP_UNSAFE_PATH", key, "absolute path or '..' in member name"))
                for sev, rule, detail in path_findings(key, "zenodo"):
                    findings.append(Finding(sev, rule, key, detail))
                if info.file_size <= 5 * MiB:
                    head = zf.open(info).read(8192)
                    if looks_text(member, head):
                        text = zf.read(info).decode("utf-8", "replace")
                        scan_text(text, key, findings, allowed_emails=args.allowed_email_set,
                                  local_path_severity="REVIEW", deferred=args.deferred)
            if manifest is not None:
                listed = manifest.get(name)
                if listed is None:
                    findings.append(Finding("FAIL", "MANIFEST_MISMATCH", name, "archive not described in MANIFEST.csv"))
                elif listed != names:
                    findings.append(Finding("FAIL", "MANIFEST_MISMATCH", name,
                                            f"{len(names - listed)} members not in MANIFEST.csv, "
                                            f"{len(listed - names)} listed but absent"))
            if args.test_zips:
                bad = zf.testzip()
                if bad:
                    findings.append(Finding("FAIL", "ZIP_CRC", f"{name}::{bad}", "CRC check failed"))
    if manifest is not None:
        for archive in sorted(set(manifest) - set(files)):
            findings.append(Finding("FAIL", "MANIFEST_MISMATCH", archive, "listed in MANIFEST.csv but absent"))
    return {"files": len(files), "bytes": total, "archive_members": members_total}


# ---------------------------------------------------------------- report

def print_report(mode: str, root: str, stats: Dict[str, object], findings: List[Finding]) -> int:
    block = blocking(findings)
    acknowledged = [f for f in findings if f.allowed_reason]
    info = [f for f in findings if f.severity == "INFO"]
    print(f"release_audit  mode={mode}  root={root}")
    print("  " + "  ".join(f"{k}={v}" for k, v in stats.items()))

    def show(title: str, items: List[Finding]) -> None:
        if not items:
            return
        print(f"\n{title} ({len(items)})")
        for f in sorted(items, key=lambda x: (x.severity, x.rule, x.path, x.line or 0)):
            loc = f"{f.path}:{f.line}" if f.line else f.path
            extra = f"  [allowed: {f.allowed_reason}]" if f.allowed_reason else ""
            print(f"  [{f.severity}] {f.rule}  {loc}  {f.detail}{extra}")

    show("Must fix", [f for f in block if f.severity == "FAIL"])
    show("Review (fix, or acknowledge in the allowlist)", [f for f in block if f.severity == "REVIEW"])
    show("Acknowledged in the allowlist", acknowledged)
    show("Information", info)
    status = "PASS" if not block else "NOT READY"
    print(f"\nResult: {status}  ({len(block)} blocking, {len(acknowledged)} acknowledged, {len(info)} info)")
    return 0 if not block else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-publication audit for the ExcaScene release.")
    ap.add_argument("--mode", choices=["github", "zenodo"], required=True)
    ap.add_argument("--root", required=True, help="repository folder (github) or deposit folder (zenodo)")
    ap.add_argument("--allowlist", help="default: <root>/.release-audit-allow.txt")
    ap.add_argument("--defer-placeholders", default="",
                    help="comma-separated placeholder names reported as INFO instead of FAIL "
                         "(only for reviews before the DOIs exist)")
    ap.add_argument("--allowed-emails", default=",".join(sorted(DEFAULT_ALLOWED_EMAILS)))
    ap.add_argument("--max-files", type=int, default=100, help="Zenodo files per record (default 100)")
    ap.add_argument("--quota-gb", type=float, default=50.0, help="Zenodo size per record in GB (default 50)")
    ap.add_argument("--skip-hash", action="store_true", help="zenodo: check presence only, not SHA-256")
    ap.add_argument("--test-zips", action="store_true", help="zenodo: read every archive and check CRCs")
    ap.add_argument("--all-files", action="store_true",
                    help="github: audit every file instead of what git would commit")
    ap.add_argument("--json", help="also write findings to this JSON file")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"not a folder: {root}", file=sys.stderr)
        return 2
    args.deferred = {n.strip() for n in args.defer_placeholders.split(",") if n.strip()}
    args.allowed_email_set = {e.strip().lower() for e in args.allowed_emails.split(",") if e.strip()}
    allow_path = args.allowlist or os.path.join(root, ".release-audit-allow.txt")
    allowlist = load_allowlist(allow_path)

    findings: List[Finding] = []
    if args.mode == "github":
        stats = audit_github(root, args, findings)
    else:
        stats = audit_zenodo(root, args, findings)
    apply_allowlist(findings, allowlist)
    code = print_report(args.mode, root, stats, findings)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"mode": args.mode, "root": root, "stats": stats,
                       "result": "PASS" if code == 0 else "NOT READY",
                       "findings": [asdict(f) for f in findings]}, fh, indent=2)
    return code


if __name__ == "__main__":
    sys.exit(main())
