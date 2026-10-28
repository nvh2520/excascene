"""Targeted custody check of reporting inputs; never traverses raw research data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'AFAST_reporting_closure/records'
run = ROOT/'external_validation_run_EV1_AFAST'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

if len(sys.argv) > 1 and sys.argv[1] == 'verify':
    baseline = json.loads((OUT/'preserved_reporting_inputs_before.json').read_text())
    changed = [r['path'] for r in baseline if not (ROOT/r['path']).is_file() or sha(ROOT/r['path']) != r['sha256']]
    result = {'checked_files':len(baseline), 'changed':changed,
              'scope':'Targeted reporting inputs and original compact ZIP; not a disk-wide audit.',
              'status':'MATCHED' if not changed else 'INTEGRITY_BLOCKER'}
    (OUT/'preserved_reporting_inputs_after.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert not changed
else:
    files = set()
    for directory in [ROOT/'paper_v5_1', ROOT/'external_validation_design',run/'code',run/'analysis',run/'metrics',run/'records',run/'resources',run/'manifests']:
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(run.glob('*.md'))
    files.update([run/'EV1_AFAST_evidence.zip',run/'final_status.json',ROOT/'revision_v5/validation_run/primary_results.csv',ROOT/'revision_v5/readiness_v5_1/author_response_form.md',ROOT/'ExcaScene_AFAST_Integration_Execution_Prompt.md',ROOT/'ExcaScene_AFAST_Review_Findings.md'])
    rows = [{'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size, 'sha256':sha(p)} for p in sorted(files)]
    (OUT/'preserved_reporting_inputs_before.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'original_zip_sha256':sha(run/'EV1_AFAST_evidence.zip')}))
