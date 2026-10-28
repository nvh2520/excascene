"""Create a local private review bundle from explicit small-artifact allowlists.

No original ZIP, raw sensor/latent/prediction arrays, model tensors or native
binaries are copied. Original scalar exports remain byte-identical.
"""
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
C = ROOT/'AFAST_reporting_closure'
R = ROOT/'external_validation_run_EV1_AFAST'
EXCLUDED = ['Raw generated sensor arrays, latent arrays and neural prediction arrays',
            'All model checkpoints/tensors (initial/current/final .pt files)',
            'Compiled native .so binaries and the runtime/dependency environment',
            'Historical private annotations, author accounts and unrelated correspondence',
            'The original immutable EV1_AFAST_evidence.zip (digest reference only)',
            'Full original historical raw datasets and training/inference environment']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    files=set()
    for base in [ROOT/'paper_v5_2', C]:
        for p in base.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(base)
            if any(x in rel.parts for x in ['__pycache__','renders','package_verification']):continue
            if p.suffix in ['.zip','.aux','.log','.out','.toc','.spl','.blg']:continue
            if p.name in ['private_review_package_manifest.json','private_review_package_receipt.json','final_delivery_receipt.json','final_resource_totals.json']:continue
            files.add(p)
    for directory in ['analysis','metrics','code','source/excascene_snapshot','source/design_original']:
        files.update(p for p in (R/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.so','.pyc'])
    for pattern in ['checkpoints/*/seed*/history.json','checkpoints/*/seed*/completion.json']:
        files.update(R.glob(pattern))
    for name in ['AFAST_results.md','frozen_protocol.md','amendment_EV1_AFAST.md','adapter_record.md','manuscript_impact_note.md','evidence_README.md','source/bevfusion_original.tar.gz','source/LICENSE_BEVFusion.txt','source/selectors_v5_original.py','source/generator_v5_original.py','records/upstream_compatibility.diff','records/all_six_final_checkpoints.json','records/holdout_prediction_release.json','records/authorized_execution_prompt.md','manifests/execution_lock.json','manifests/analysis_lock.json','reports/deployment_timing.json']:
        files.add(R/name)
    assert all(p.is_file() for p in files)
    assert not any(p.suffix in ['.pt','.so','.zip'] for p in files)
    rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    assert sum(r['bytes'] for r in rows)<500_000_000, 'Unexpectedly large compact review payload'
    original_digest='0f5e2a4ebd3bd415efb5c153b88e68485e91c4df481e28121616a434e4846ec4'
    manifest=dict(package='ExcaScene v5.2 PRIVATE author-review package',created_unix=time.time(),
                  original_afast_archive_sha256=original_digest,intentionally_excluded=EXCLUDED,
                  scope='Editable manuscript/PDFs and scalar evidence. No independent training/inference reproduction.',
                  resource_note='Historical AFAST final receipts are included. Reporting resources at package freeze are included; the actual new package-verification costs and final totals are attached beside this archive.',
                  included_payload=rows)
    m=C/'records/private_review_package_manifest.json'
    m.write_text(json.dumps(manifest,indent=2)+'\n')
    output=C/'ExcaScene_v5_2_PRIVATE_author_review.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files|{m}):z.write(p,str(p.relative_to(ROOT)))
    receipt=dict(status='PRIVATE_PACKAGE_WRITTEN',path=str(output.relative_to(ROOT)),bytes=output.stat().st_size,
                 sha256=sha(output),payload_files=len(rows),original_afast_archive_sha256=original_digest,
                 public_release_authorized=False,submission_authorized=False)
    (C/'records/private_review_package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
