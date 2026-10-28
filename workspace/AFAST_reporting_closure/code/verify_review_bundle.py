"""Portable, read-only custody verifier (stdlib only; no models or GPU imports).

Usage after extraction: python AFAST_reporting_closure/code/verify_review_bundle.py --root .
Optional --original-archive checks an independently supplied original immutable ZIP.
This verifies available bytes, not training, neural inference, or excluded artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--original-archive', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    manifest = json.loads((root/'AFAST_reporting_closure/records/private_review_package_manifest.json').read_text())
    failures = []
    for row in manifest['included_payload']:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root):
            failures.append({'path':row['path'],'reason':'path escapes package'})
        elif not path.is_file():
            failures.append({'path':row['path'],'reason':'missing'})
        elif path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            failures.append({'path':row['path'],'reason':'digest/size mismatch'})
    provenance=json.loads((root/'AFAST_reporting_closure/scalar_source_provenance.json').read_text())
    for row in provenance['files']:
        path=root/provenance['source_root_name']/row['path']
        if not path.is_file() or sha(path)!=row['sha256']:
            failures.append({'path':str(path.relative_to(root)),'reason':'scalar replay dependency missing or changed'})
    archive_status = 'REFERENCE_ONLY_NOT_SUPPLIED'
    if a.original_archive:
        archive_status = 'MATCHED' if sha(a.original_archive) == manifest['original_afast_archive_sha256'] else 'MISMATCH'
        if archive_status == 'MISMATCH':
            failures.append({'path':str(a.original_archive),'reason':'original archive digest mismatch'})
    report = dict(status='AVAILABLE_PAYLOAD_VERIFIED' if not failures else 'INTEGRITY_BLOCKER',
                  checked_payload_files=len(manifest['included_payload']), failures=failures,
                  scalar_replay_dependencies_checked=len(provenance['files']),
                  original_archive=archive_status,
                  intentionally_excluded=manifest['intentionally_excluded'],
                  limits='File integrity only. Scalar replay is separate. No independent training, neural inference, checkpoint or native-binary reproduction is claimed.')
    print(json.dumps(report, indent=2))
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())
