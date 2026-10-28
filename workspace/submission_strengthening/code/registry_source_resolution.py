"""Resolve registry dependencies within the candidate; do not re-audit history."""
from pathlib import Path
import csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'submission_strengthening';C=ROOT/'release_candidate'
registry=json.loads((S/'publication_numbers_submission_candidate.json').read_text())
index={}
for r in csv.DictReader((C/'payload_manifest.csv').open()):index.setdefault(r['sha256'],[]).append('release_candidate/'+r['path'])
for p in (ROOT/'paper_submission_candidate/data').rglob('*'):
    if p.is_file():index.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(),[]).append(str(p.relative_to(ROOT)))
allowed={'resource_summary.json','bundle_seal.json','branch_gates.json','accepted_articulation_mix.csv','legacy_camera_headline_old_new.csv','validation_protocol_amended.md','precision_report.md'}
private={'author_reply_record.json','author_statement_status.json'}
entries=list(csv.DictReader((C/'allowlist_sources.csv').open()))
rows=[]
for record in registry['records']:
    digest=record.get('source_sha256');source=record.get('source_file','')
    if not digest:continue
    if digest not in index and record in registry['records'][:565] and Path(source).name in allowed:
        src=ROOT/record['package_source_file'];assert hashlib.sha256(src.read_bytes()).hexdigest()==digest
        dest=C/'statistics/submission_candidate/registry_sources'/digest/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        index[digest]=[str(dest.relative_to(ROOT))]
        entries.append(dict(path=str(dest.relative_to(C)),source=str(src.relative_to(ROOT)),bytes=dest.stat().st_size,sha256=digest,category='explicit_registry_scientific_provenance'))
    if digest not in index and source.startswith('submission_strengthening/'):
        p=ROOT/source;assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
        index[digest]=[source]
    if digest in index:status='INCLUDED_IN_PRIVATE_HANDOFF';paths=';'.join(index[digest])
    elif Path(source).name in private:status='PRIVATE_ATTRIBUTED_AUTHOR_SOURCE_NOT_BUNDLED';paths='preserved original: '+record.get('package_source_file',source)
    else:raise RuntimeError(('Unresolved scientific registry source',record['id'],source,digest))
    rows.append(dict(record_id=record['id'],original_source=source,sha256=digest,status=status,package_paths_or_retained_locator=paths))
with (C/'allowlist_sources.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
out=S/'registry_source_resolution.csv'
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
shutil.copy2(out,C/'statistics/submission_candidate'/out.name)
print(json.dumps({'registry_records_resolved':len(rows),'scientific_source_missing':0,'private_author_source_records_not_bundled':sum(r['status'].startswith('PRIVATE') for r in rows)},indent=2))
